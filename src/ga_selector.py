import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np

# Authoritative GA configuration used by Stage 3.
GA_CONFIG = {
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

def ga_fitness(cv_macro_f1, n_selected, n_features=27, penalty=0.01):
    """Macro-F1 validation score with complexity penalty."""
    return float(cv_macro_f1 - penalty * (n_selected / n_features))

def initialize_population(rng, n_features=27, population_size=6, min_selected=3):
    pop=[]
    for _ in range(population_size):
        m=rng.random(n_features)<0.5
        while int(m.sum()) < min_selected:
            m[rng.integers(n_features)] = True
        pop.append(m)
    return pop

def tournament_selection(population, fitness_values, k=3, rng=None):
    """Standard tournament selection."""
    rng = rng or np.random.default_rng()
    k=min(k,len(population))
    idx=rng.choice(len(population),size=k,replace=False)
    winner=idx[np.argmax([fitness_values[i] for i in idx])]
    return population[winner].copy()

def crossover(parent1, parent2, rng=None, probability=0.80):
    rng=rng or np.random.default_rng()
    if rng.random()>probability:
        return parent1.copy(),parent2.copy()
    point=int(rng.integers(1,len(parent1)-1))
    return (np.r_[parent1[:point],parent2[point:]],
            np.r_[parent2[:point],parent1[point:]])

def mutation(chromosome, rng=None, probability=0.04, min_selected=3):
    rng=rng or np.random.default_rng()
    child=chromosome.copy()
    mask=rng.random(len(child))<probability
    child[mask]=~child[mask]
    while int(child.sum())<min_selected:
        child[rng.integers(len(child))]=True
    return child

def run_ga(population_size=6, generations=3, tournament_k=3, seed=42,
           mutation_probability=0.04, crossover_probability=0.80, elitism=2):
    """Configuration record for the authoritative GA implementation."""
    return {"chromosome_length":27,"population_size":population_size,"generations":generations,
            "elitism":elitism,"tournament_k":tournament_k,"mutation_probability":mutation_probability,
            "crossover_probability":crossover_probability,"seed":seed}

def summarize_seeds(records):
    return {"n_seeds":len(records),"min_selected":min(r["n_selected"] for r in records),
            "max_selected":max(r["n_selected"] for r in records)}
