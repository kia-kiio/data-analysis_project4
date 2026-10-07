"""
Project: Steel Plate Fault Type Classification
Module: src/genetic_feature_selection.py
Description: Production-grade Genetic Algorithm for feature selection with
complexity-penalized Macro F1 fitness, multi-seed stability, and pairwise Jaccard similarity.
Architect: Kiana Sarkari
"""

import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils import RESULTS_DIR, FEATURE_NAMES

# Authoritative GA configuration
DEFAULT_GA_CONFIG = {
    "chromosome_length": 27,
    "population_size": 6,
    "generations": 3,
    "elitism": 2,
    "tournament_k": 3,
    "crossover_probability": 0.80,
    "mutation_probability": 0.04,
    "min_selected_features": 3,
    "fitness_penalty": 0.01,
    "cv_splits": 3,
    "cv_seed": 123,
    "seeds": (11, 22, 33),
}

def ga_fitness_score(cv_macro_f1, n_selected, n_features=27, penalty=0.01):
    """
    Complexity-penalized fitness calculation:
    Fitness = CV_Macro_F1 - penalty * (n_selected / total_features)
    """
    return float(cv_macro_f1 - penalty * (n_selected / n_features))

class GeneticFeatureSelector:
    """
    Genetic Algorithm for Feature Selection on Imbalanced Industrial Data.
    Operates strictly within Training data using internal Stratified Cross-Validation.
    """
    def __init__(
        self,
        chromosome_length=27,
        population_size=6,
        generations=3,
        elitism=2,
        tournament_k=3,
        crossover_probability=0.80,
        mutation_probability=0.04,
        min_selected_features=3,
        fitness_penalty=0.01,
        cv_splits=3,
        cv_seed=123,
        estimator_type="logistic",
        estimator_params=None,
        random_state=42
    ):
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.generations = generations
        self.elitism = elitism
        self.tournament_k = tournament_k
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.min_selected_features = min_selected_features
        self.fitness_penalty = fitness_penalty
        self.cv_splits = cv_splits
        self.cv_seed = cv_seed
        self.estimator_type = estimator_type
        self.estimator_params = estimator_params or {}
        self.random_state = random_state
        
        self.best_chromosome_ = None
        self.best_fitness_ = -np.inf
        self.best_cv_macro_f1_ = 0.0
        self.history_ = []
        self.cache_ = {}
        self.runtime_sec_ = 0.0

    def _get_estimator_pipeline(self):
        """Constructs the evaluator pipeline for fitness evaluation."""
        if self.estimator_type == "logistic":
            C = self.estimator_params.get("C", 1.0)
            return Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", C=C, solver="lbfgs", random_state=42))
            ])
        elif self.estimator_type == "random_forest":
            n_est = self.estimator_params.get("n_estimators", 50)
            return RandomForestClassifier(
                n_estimators=n_est,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=1
            )
        else:
            raise ValueError(f"Unknown estimator_type: {self.estimator_type}")

    def evaluate_chromosome(self, chromosome, X_train, y_train, cv):
        """
        Evaluates a single binary chromosome using Stratified CV on X_train.
        Caches results to avoid redundant evaluations of identical individuals.
        """
        key = tuple(chromosome)
        if key in self.cache_:
            return self.cache_[key]
            
        selected_indices = np.flatnonzero(chromosome)
        n_selected = len(selected_indices)
        
        if n_selected < self.min_selected_features:
            return -1.0, 0.0
            
        feats = [FEATURE_NAMES[i] for i in selected_indices]
        pipe = self._get_estimator_pipeline()
        
        scores = cross_validate(pipe, X_train[feats], y_train, cv=cv, scoring="f1_macro", n_jobs=1)
        mean_macro_f1 = float(scores["test_score"].mean())
        fitness = ga_fitness_score(mean_macro_f1, n_selected, self.chromosome_length, self.fitness_penalty)
        
        self.cache_[key] = (fitness, mean_macro_f1)
        return fitness, mean_macro_f1

    def initialize_population(self, rng):
        """Generates initial population with minimum feature constraint."""
        population = []
        for _ in range(self.population_size):
            mask = rng.random(self.chromosome_length) < 0.5
            while int(mask.sum()) < self.min_selected_features:
                mask[rng.integers(self.chromosome_length)] = True
            population.append(mask)
        return population

    def tournament_selection(self, population, fitness_values, rng):
        """Tournament selection operator."""
        k = min(self.tournament_k, len(population))
        indices = rng.choice(len(population), size=k, replace=False)
        winner_idx = indices[np.argmax([fitness_values[i] for i in indices])]
        return population[winner_idx].copy()

    def crossover(self, parent1, parent2, rng):
        """Single-point crossover with probability crossover_probability."""
        if rng.random() > self.crossover_probability:
            return parent1.copy(), parent2.copy()
        point = int(rng.integers(1, self.chromosome_length - 1))
        child1 = np.r_[parent1[:point], parent2[point:]]
        child2 = np.r_[parent2[:point], parent1[point:]]
        return child1, child2

    def mutate(self, chromosome, rng):
        """Bit-flip mutation with probability mutation_probability."""
        child = chromosome.copy()
        mask = rng.random(len(child)) < self.mutation_probability
        child[mask] = ~child[mask]
        while int(child.sum()) < self.min_selected_features:
            child[rng.integers(len(child))] = True
        return child

    def fit(self, X_train, y_train, seed=None):
        """
        Executes the GA evolution over the specified number of generations.
        """
        run_seed = seed if seed is not None else self.random_state
        rng = np.random.default_rng(run_seed)
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.cv_seed)
        
        t0 = time.perf_counter()
        population = self.initialize_population(rng)
        self.history_ = []
        best_overall = None
        
        for g in range(self.generations):
            scored = []
            for indiv in population:
                fit_val, f1_val = self.evaluate_chromosome(indiv, X_train, y_train, cv)
                scored.append((fit_val, f1_val, indiv.copy()))
                
            scored.sort(key=lambda x: x[0], reverse=True)
            current_best = scored[0]
            if best_overall is None or current_best[0] > best_overall[0]:
                best_overall = current_best
                
            self.history_.append({
                "seed": run_seed,
                "generation": g,
                "best_fitness": current_best[0],
                "best_cv_macro_f1": current_best[1],
                "n_features": int(current_best[2].sum())
            })
            
            # Elitism: retain top individuals
            elites = [s[2].copy() for s in scored[:self.elitism]]
            fitness_values = [s[0] for s in scored]
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                p1 = self.tournament_selection([s[2] for s in scored], fitness_values, rng)
                p2 = self.tournament_selection([s[2] for s in scored], fitness_values, rng)
                child, _ = self.crossover(p1, p2, rng)
                child = self.mutate(child, rng)
                new_population.append(child)
                
            population = new_population
            
        self.runtime_sec_ = time.perf_counter() - t0
        self.best_fitness_ = best_overall[0]
        self.best_cv_macro_f1_ = best_overall[1]
        self.best_chromosome_ = best_overall[2]
        
        return self

def run_multi_seed_ga(X_train, y_train, seeds=(11, 22, 33), config=None, estimator_params=None):
    """
    Executes GA across at least 3 independent seeds and calculates full stability metrics:
    - Selection frequency per feature
    - Pairwise Jaccard similarity across subsets
    - Mean and standard deviation of fitness and CV Macro F1
    """
    cfg = config or DEFAULT_GA_CONFIG
    results = []
    histories = []
    masks = []
    
    for seed in seeds:
        ga = GeneticFeatureSelector(
            chromosome_length=cfg["chromosome_length"],
            population_size=cfg["population_size"],
            generations=cfg["generations"],
            elitism=cfg["elitism"],
            tournament_k=cfg["tournament_k"],
            crossover_probability=cfg["crossover_probability"],
            mutation_probability=cfg["mutation_probability"],
            min_selected_features=cfg["min_selected_features"],
            fitness_penalty=cfg["fitness_penalty"],
            cv_splits=cfg["cv_splits"],
            cv_seed=cfg["cv_seed"],
            estimator_params=estimator_params or {},
            random_state=seed
        )
        ga.fit(X_train, y_train, seed=seed)
        
        mask = ga.best_chromosome_
        selected_feats = [FEATURE_NAMES[i] for i in np.flatnonzero(mask)]
        masks.append(mask)
        histories.extend(ga.history_)
        
        results.append({
            "seed": seed,
            "best_fitness": ga.best_fitness_,
            "cv_macro_f1": ga.best_cv_macro_f1_,
            "n_selected": len(selected_feats),
            "selected_features": "|".join(selected_feats),
            "runtime_sec": ga.runtime_sec_
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "ga_feature_selection_results.csv", index=False)
    
    # Required genetic_selection.csv format
    gen_df = pd.DataFrame({
        "Seed": res_df["seed"],
        "Selected features": res_df["selected_features"],
        "Number of features": res_df["n_selected"],
        "Fitness": res_df["best_fitness"],
        "CV Macro F1": res_df["cv_macro_f1"],
        "Runtime": res_df["runtime_sec"]
    })
    gen_df.to_csv(RESULTS_DIR / "genetic_selection.csv", index=False)
    
    # Save configuration and convergence
    (RESULTS_DIR / "ga_configuration.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    pd.DataFrame(histories).to_csv(RESULTS_DIR / "ga_convergence.csv", index=False)
    
    # Feature frequency across seeds
    freq_rows = []
    for idx, feat in enumerate(FEATURE_NAMES):
        count = sum(m[idx] for m in masks)
        freq_rows.append({
            "feature": feat,
            "selection_count": count,
            "selection_frequency": count / len(seeds),
            "stability_class": "Stable" if count == len(seeds) else ("Moderate" if count > 0 else "Excluded")
        })
    freq_df = pd.DataFrame(freq_rows).sort_values("selection_frequency", ascending=False)
    freq_df.to_csv(RESULTS_DIR / "ga_feature_frequency.csv", index=False)
    
    # Pairwise subset overlap and Jaccard similarity
    overlap_rows = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            inter = int(np.logical_and(masks[i], masks[j]).sum())
            union = int(np.logical_or(masks[i], masks[j]).sum())
            jaccard = inter / union if union > 0 else 1.0
            overlap_rows.append({
                "seed_a": seeds[i],
                "seed_b": seeds[j],
                "intersection": inter,
                "union": union,
                "jaccard": jaccard
            })
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(RESULTS_DIR / "ga_subset_overlap.csv", index=False)
    
    return res_df, freq_df, overlap_df, masks
