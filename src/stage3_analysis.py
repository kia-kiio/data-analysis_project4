from pathlib import Path
import json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from ga_selector import GA_CONFIG, initialize_population, tournament_selection, crossover, mutation, ga_fitness
warnings.filterwarnings('ignore')

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/processed_dataset.csv'; RES=ROOT/'results'; FIG=ROOT/'figures'
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
classes=['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']
TARGET='Class'; FEATURES=pd.read_csv(RES/'data_dictionary.csv').query("role=='feature'")['column'].tolist()

df=pd.read_csv(DATA); X=df[FEATURES].copy(); y=df[TARGET].copy()
split=pd.read_csv(RES/'split_indices.csv')
# split_indices may store row_index + split; support either layout
if {'row_index','split'}.issubset(split.columns):
    tr=split.loc[split.split=='Train','row_index'].astype(int).to_numpy(); va=split.loc[split.split=='Validation','row_index'].astype(int).to_numpy(); te=split.loc[split.split=='Test','row_index'].astype(int).to_numpy()
else: raise ValueError('Unexpected split_indices.csv schema')
Xtr,ytr=X.loc[tr],y.loc[tr]; Xva,yva=X.loc[va],y.loc[va]; Xte,yte=X.loc[te],y.loc[te]
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

def metrics(ytrue,ypred):
    return {'accuracy':accuracy_score(ytrue,ypred),'balanced_accuracy':balanced_accuracy_score(ytrue,ypred),'macro_f1':f1_score(ytrue,ypred,average='macro'),'weighted_f1':f1_score(ytrue,ypred,average='weighted'),'macro_precision':precision_score(ytrue,ypred,average='macro',zero_division=0),'macro_recall':recall_score(ytrue,ypred,average='macro',zero_division=0)}

def perclass(ytrue,ypred):
    p=precision_score(ytrue,ypred,labels=classes,average=None,zero_division=0); r=recall_score(ytrue,ypred,labels=classes,average=None,zero_division=0); f=f1_score(ytrue,ypred,labels=classes,average=None,zero_division=0)
    out=pd.DataFrame({'class':classes,'precision':p,'recall':r,'f1':f})
    out['support']=[int((ytrue==c).sum()) for c in classes]
    out['false_negatives']=[int(((ytrue==c)&(ypred!=c)).sum()) for c in classes]
    return out

def cv_eval(name, pipe, X0=Xtr, y0=ytr):
    t=time.perf_counter(); sc=cross_validate(pipe,X0,y0,cv=cv,scoring={'macro_f1':'f1_macro','bal_acc':'balanced_accuracy'},n_jobs=1); dt=time.perf_counter()-t
    return {'method':name,'cv_macro_f1_mean':sc['test_macro_f1'].mean(),'cv_macro_f1_std':sc['test_macro_f1'].std(ddof=1),'cv_balanced_accuracy_mean':sc['test_bal_acc'].mean(),'runtime_sec':dt}

# Hyperparameter search on Train only; budgets: Logistic 5 candidates, Random Forest 3 candidates, 5-fold Stratified CV.
search_cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
logit_base=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=4000,class_weight='balanced'))])
logit_grid={'model__C':[0.1,0.5,1.0,2.0,5.0], 'model__solver':['lbfgs']}
logit_search=RandomizedSearchCV(logit_base,logit_grid,cv=search_cv,scoring='f1_macro',n_jobs=1,refit=True,return_train_score=True,n_iter=5,random_state=2026)
t0=time.perf_counter(); logit_search.fit(Xtr,ytr); logit_search_runtime=time.perf_counter()-t0
rf_base=RandomForestClassifier(class_weight='balanced_subsample',random_state=42,n_jobs=1)
rf_grid={'n_estimators':[50,80], 'min_samples_leaf':[1,2,4], 'max_features':['sqrt',0.5]}
rf_search=RandomizedSearchCV(rf_base,rf_grid,cv=search_cv,scoring='f1_macro',n_jobs=1,refit=True,return_train_score=True,n_iter=3,random_state=2026)
t0=time.perf_counter(); rf_search.fit(Xtr,ytr); rf_search_runtime=time.perf_counter()-t0
hp_rows=[]
for label,search,runtime in [('Logistic_Interpretable',logit_search,logit_search_runtime),('RandomForest_Nonlinear',rf_search,rf_search_runtime)]:
    hp_rows.append({'model':label,'best_cv_macro_f1':search.best_score_,'best_params':json.dumps(search.best_params_,sort_keys=True),'n_candidates':len(search.cv_results_['params']), 'search_type':'RandomizedSearchCV_5fold', 'search_seed':2026,'runtime_sec':runtime})
pd.DataFrame(hp_rows).to_csv(RES/'hyperparameter_search.csv',index=False)

# Default-vs-tuned comparison on the same Train-only 5-fold CV protocol.
default_gap_rows=[]
for label, default_est, tuned_est in [
    ('Logistic_Interpretable',
     Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=4000,class_weight='balanced'))]),
     logit_search.best_estimator_),
    ('RandomForest_Nonlinear',
     RandomForestClassifier(class_weight='balanced_subsample',random_state=42,n_jobs=1),
     rf_search.best_estimator_)
]:
    for status, est in [('default',default_est),('tuned',tuned_est)]:
        sc=cross_validate(est,Xtr,ytr,cv=cv,scoring='f1_macro',return_train_score=True,n_jobs=1)
        default_gap_rows.append({
            'model':label,'status':status,
            'train_macro_f1_mean':float(sc['train_score'].mean()),
            'train_macro_f1_std':float(sc['train_score'].std(ddof=1)),
            'cv_macro_f1_mean':float(sc['test_score'].mean()),
            'cv_macro_f1_std':float(sc['test_score'].std(ddof=1)),
            'train_cv_gap':float(sc['train_score'].mean()-sc['test_score'].mean())
        })
pd.DataFrame(default_gap_rows).to_csv(RES/'hyperparameter_default_vs_tuned.csv',index=False)

# Baselines
maj=ytr.value_counts().idxmax(); yp=np.repeat(maj,len(yva)); m=metrics(yva,yp)
base_rows=[{'method':'Majority_Baseline',**m,'runtime_sec':0.0}]
logit=logit_search.best_estimator_
rf=rf_search.best_estimator_
for name,model in [('Logistic_Interpretable',logit),('RandomForest_Nonlinear',rf)]:
    t=time.perf_counter(); model.fit(Xtr,ytr); pred=model.predict(Xva); dt=time.perf_counter()-t; base_rows.append({'method':name,**metrics(yva,pred),'runtime_sec':dt});
    if name=='Logistic_Interpretable': perclass(yva,pred).to_csv(RES/'baseline_logistic_per_class.csv',index=False)
base=pd.DataFrame(base_rows); base.to_csv(RES/'baseline_models.csv',index=False)
# Per-class validation metrics for all baseline models
bpc=[]
for name,model in [('Logistic_Interpretable',logit),('RandomForest_Nonlinear',rf)]:
    pr=model.predict(Xva); z=perclass(yva,pr); z.insert(0,'model',name); bpc.append(z)
pd.concat(bpc,ignore_index=True).to_csv(RES/'baseline_per_class_metrics.csv',index=False)

# Robust strategies: standard vs robust scaling and train-fitted winsorization+robust scaling
class TrainWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99): self.lower=lower; self.upper=upper
    def fit(self, X, y=None):
        A=np.asarray(X, dtype=float); self.lo_=np.nanpercentile(A, self.lower*100, axis=0); self.hi_=np.nanpercentile(A, self.upper*100, axis=0); return self
    def transform(self, X): return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)
clipper=TrainWinsorizer(0.01,0.99)
robust_pipes={
 'StandardScaler_Logistic':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))]),
 'RobustScaler_Logistic':Pipeline([('scale',RobustScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))]),
 'Winsorize1to99_RobustScaler_Logistic':Pipeline([('clip',clipper),('scale',RobustScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))])}
rr=[]
for name,p in robust_pipes.items():
    t=time.perf_counter(); p.fit(Xtr,ytr); pred=p.predict(Xva); dt=time.perf_counter()-t; mm=metrics(yva,pred); 
    pc=perclass(yva,pred); row={'strategy':name,**mm,'runtime_sec':dt}
    for _,z in pc.iterrows():
        row[f'recall_{z["class"]}']=float(z['recall']); row[f'precision_{z["class"]}']=float(z['precision']); row[f'f1_{z["class"]}']=float(z['f1'])
    rr.append(row)
pd.DataFrame(rr).to_csv(RES/'robustness_comparison.csv',index=False)

# Noise sensitivity is evaluated on the FINAL selected model/subset only.
# Model fitting uses Train; perturbation evaluation uses Validation; Test is excluded.
# Filter selection: deterministic train-only correlation filter abs(r)>=.90; drop second feature in ordered pairs
corr=Xtr.corr(numeric_only=True); removed=set(); pairs=[]
for i,a in enumerate(FEATURES):
    for b in FEATURES[i+1:]:
        r=corr.loc[a,b]
        if abs(r)>=0.90:
            keep=a if a not in removed else (b if b not in removed else a)
            drop=b if keep==a else a
            if keep not in removed and drop not in removed: removed.add(drop)
            pairs.append((a,b,float(r),keep,drop))
filter_features=[f for f in FEATURES if f not in removed]
filter_df=pd.DataFrame(pairs,columns=['feature_1','feature_2','correlation','kept','removed'])
filter_pipe=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))])
t=time.perf_counter(); filter_pipe.fit(Xtr[filter_features],ytr); pred=filter_pipe.predict(Xva[filter_features]); filter_runtime=time.perf_counter()-t
filter_res=pd.DataFrame([{'selection_method':'correlation_filter_abs_0.90','threshold':0.90,'selected_features':'|'.join(filter_features),'n_selected':len(filter_features),'validation_macro_f1':f1_score(yva,pred,average='macro'),'validation_balanced_accuracy':balanced_accuracy_score(yva,pred),'runtime_sec':filter_runtime}]); filter_res.to_csv(RES/'filter_feature_selection.csv',index=False)

# GA on Train/CV. Authoritative operators/configuration are imported from ga_selector.py.
# Fitness = mean 3-fold CV Macro F1 - 0.01*(n_selected/27); minimum 3 features.
alpha=GA_CONFIG["fitness_penalty"]; ga_cv=StratifiedKFold(n_splits=GA_CONFIG["cv_splits"],shuffle=True,random_state=GA_CONFIG["cv_seed"]); cache={}

def fitness(mask):
    key=tuple(mask)
    if key in cache:return cache[key]
    idx=np.flatnonzero(mask)
    if len(idx)<GA_CONFIG["min_selected_features"]:return -1.0,0.0
    feats=[FEATURES[i] for i in idx]
    pipe=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced', C=logit_search.best_params_.get('model__C',1.0), solver='lbfgs'))])
    sc=cross_validate(pipe,Xtr[feats],ytr,cv=ga_cv,scoring='f1_macro',n_jobs=1)
    mf=float(sc['test_score'].mean()); fit=ga_fitness(mf,len(idx),len(FEATURES),alpha); cache[key]=(fit,mf); return fit,mf

def run_ga(seed):
    rng=np.random.default_rng(seed); pop=initialize_population(rng,GA_CONFIG["chromosome_length"],GA_CONFIG["population_size"],GA_CONFIG["min_selected_features"])
    history=[]; start=time.perf_counter(); global_best=None
    for g in range(GA_CONFIG["generations"]):
        scored=[(*fitness(m),m.copy()) for m in pop]; scored.sort(key=lambda z:z[0],reverse=True); gen_best=scored[0]
        if global_best is None or gen_best[0]>global_best[0]: global_best=gen_best
        history.append({'seed':seed,'generation':g,'best_fitness':global_best[0],'best_cv_macro_f1':global_best[1],'n_features':int(global_best[2].sum())})
        elites=[s[2].copy() for s in scored[:GA_CONFIG["elitism"]]]
        fitness_values=[s[0] for s in scored]
        new=elites
        while len(new)<GA_CONFIG["population_size"]:
            a=tournament_selection([s[2] for s in scored],fitness_values,k=GA_CONFIG["tournament_k"],rng=rng)
            b=tournament_selection([s[2] for s in scored],fitness_values,k=GA_CONFIG["tournament_k"],rng=rng)
            child,_=crossover(a,b,rng=rng,probability=GA_CONFIG["crossover_probability"])
            child=mutation(child,rng=rng,probability=GA_CONFIG["mutation_probability"],min_selected=GA_CONFIG["min_selected_features"])
            new.append(child)
        pop=new
    fit,mf,mask=global_best
    return {'seed':seed,'best_fitness':fit,'cv_macro_f1':mf,'n_selected':int(mask.sum()),'selected_features':'|'.join(np.array(FEATURES)[mask]),'runtime_sec':time.perf_counter()-start},history,mask

ga_results=[]; histories=[]; masks=[]
for seed in [11,22,33]:
    r,h,m=run_ga(seed); ga_results.append(r); histories.extend(h); masks.append(m)
pd.DataFrame(ga_results).to_csv(RES/'ga_feature_selection_results.csv',index=False)
(RES/'ga_configuration.json').write_text(json.dumps(GA_CONFIG,indent=2,ensure_ascii=False))
pd.DataFrame(histories).to_csv(RES/'ga_convergence.csv',index=False)
freq=[]
for f in FEATURES: freq.append({'feature':f,'selection_count':sum(m[FEATURES.index(f)] for m in masks),'selection_frequency':sum(m[FEATURES.index(f)] for m in masks)/len(masks)})
pd.DataFrame(freq).to_csv(RES/'ga_feature_frequency.csv',index=False)
# overlap
ov=[]
for i in range(3):
 for j in range(i+1,3):
    inter=int(np.logical_and(masks[i],masks[j]).sum()); union=int(np.logical_or(masks[i],masks[j]).sum()); ov.append({'seed_a':[11,22,33][i],'seed_b':[11,22,33][j],'intersection':inter,'union':union,'jaccard':inter/union if union else 1.0})
pd.DataFrame(ov).to_csv(RES/'ga_subset_overlap.csv',index=False)

# Compare all/filter/GA using validation; GA subset per seed, aggregate mean/std
allpipe=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))]); t=time.perf_counter(); allpipe.fit(Xtr,ytr); p=allpipe.predict(Xva); allrt=time.perf_counter()-t
comp=[{'method':'All_27_Features','n_features':27,'validation_macro_f1':f1_score(yva,p,average='macro'),'balanced_accuracy':balanced_accuracy_score(yva,p),'runtime_sec':allrt,'stability':'single_fixed_set'}]
for r,m in zip(ga_results,masks):
    feats=list(np.array(FEATURES)[m]); pipe=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))]); t=time.perf_counter(); pipe.fit(Xtr[feats],ytr); pr=pipe.predict(Xva[feats]); dt=time.perf_counter()-t; comp.append({'method':f'GA_seed_{r["seed"]}','n_features':len(feats),'validation_macro_f1':f1_score(yva,pr,average='macro'),'balanced_accuracy':balanced_accuracy_score(yva,pr),'runtime_sec':dt,'stability':'see_ga_subset_overlap'})
comp.append({'method':'Filter_abs_corr_0.90','n_features':len(filter_features),'validation_macro_f1':f1_score(yva,pred,average='macro'),'balanced_accuracy':balanced_accuracy_score(yva,pred),'runtime_sec':filter_runtime,'stability':'deterministic'})
pd.DataFrame(comp).to_csv(RES/'feature_selection_comparison.csv',index=False)

# Unified model + feature-subset selection using the held-out Validation split.
# This is the authoritative selection table; Test is not used for ranking.
candidates={'All_27':FEATURES,'Filter':filter_features}
for r,m in zip(ga_results,masks):
    candidates[f'GA_{r["seed"]}']=list(np.array(FEATURES)[m])

rows=[]
logit_C=float(logit_search.best_params_.get('model__C',1.0))
rf_best=rf_search.best_params_.copy()

# Leakage-safe candidate ranking:
# feature selection (Filter/GA) is learned from Train only, so candidates are
# ranked once on the held-out Validation partition rather than re-evaluated
# with CV on the same Train data.
for model_name in ['Logistic_Interpretable','RandomForest_Nonlinear']:
    for cname,feats in candidates.items():
        if model_name=='Logistic_Interpretable':
            est=Pipeline([('scale',StandardScaler()),
                          ('model',LogisticRegression(max_iter=4000,class_weight='balanced',
                                                      C=logit_C,solver='lbfgs'))])
        else:
            est=RandomForestClassifier(class_weight='balanced_subsample',
                                       random_state=42,n_jobs=1,**rf_best)
        t0=time.perf_counter()
        est.fit(Xtr[feats],ytr)
        val_pred=est.predict(Xva[feats])
        rows.append({
            'model':model_name,
            'candidate':cname,
            'n_features':len(feats),
            'validation_macro_f1':f1_score(yva,val_pred,average='macro'),
            'validation_balanced_accuracy':balanced_accuracy_score(yva,val_pred),
            'validation_accuracy':accuracy_score(yva,val_pred),
            'runtime_sec':time.perf_counter()-t0,
            'selection_split':'Validation (held out from feature selection)'
        })

unified=pd.DataFrame(rows)
unified.to_csv(RES/'unified_model_selection_validation.csv',index=False)
chosen=unified.sort_values(['validation_macro_f1','n_features'],ascending=[False,True]).iloc[0]
chosen_model=str(chosen['model']); chosen_candidate=str(chosen['candidate']); chosen_feats=candidates[chosen_candidate]

# Keep the experiment catalog numerically consistent with the authoritative
# held-out Validation table above. In particular, GA/Filter rows must not
# label their GA fitness as if it were a held-out Validation score.
exp = pd.read_csv(RES / 'experiments.csv')
for i, row in exp.iterrows():
    model = None
    candidate = None
    if row['Experiment ID'] == 'EXP_02_Logistic_Baseline':
        model, candidate = 'Logistic_Interpretable', 'All_27'
    elif row['Experiment ID'] == 'EXP_03_RandomForest_Baseline':
        model, candidate = 'RandomForest_Nonlinear', 'All_27'
    elif row['Experiment ID'] == 'EXP_04_Logistic_Filter':
        model, candidate = 'Logistic_Interpretable', 'Filter'
    elif row['Experiment ID'] == 'EXP_05_RF_Filter':
        model, candidate = 'RandomForest_Nonlinear', 'Filter'
    elif str(row['Experiment ID']).startswith('EXP_06_GA_Seed_'):
        seed = str(row['Experiment ID']).split('_')[-1]
        model, candidate = 'Logistic_Interpretable', f'GA_{seed}'
    if model and candidate:
        hit = unified[(unified['model'] == model) & (unified['candidate'] == candidate)].iloc[0]
        exp.loc[i, 'Validation Macro F1'] = float(hit['validation_macro_f1'])
        exp.loc[i, 'Balanced Accuracy'] = float(hit['validation_balanced_accuracy'])
        exp.loc[i, 'Accuracy'] = float(hit['validation_accuracy'])
        exp.loc[i, 'Runtime'] = float(hit['runtime_sec'])
exp.to_csv(RES / 'experiments.csv', index=False)

# A dedicated, consistent feature-selection comparison uses the same
# Logistic Regression evaluator for All/Filter/GA and the held-out Validation.
fs_rows = unified[unified['model'] == 'Logistic_Interpretable'].copy()
fs_rows['method'] = fs_rows['candidate'].map(lambda x: 'All_27_Features' if x == 'All_27' else ('Filter_abs_corr_0.90' if x == 'Filter' else x))
fs_rows['n_features'] = fs_rows['n_features']
fs_rows['validation_macro_f1'] = fs_rows['validation_macro_f1']
fs_rows['balanced_accuracy'] = fs_rows['validation_balanced_accuracy']
fs_rows['runtime_sec'] = fs_rows['runtime_sec']
fs_rows['stability'] = fs_rows['candidate'].map(lambda x: 'single_fixed_set' if x == 'All_27' else ('deterministic' if x == 'Filter' else 'see_ga_subset_overlap'))
fs_rows[['method','n_features','validation_macro_f1','balanced_accuracy','runtime_sec','stability']].to_csv(RES / 'feature_selection_comparison.csv', index=False)

# Final Test: one evaluation only, after all Train/Validation selections are frozen.
Xfit=pd.concat([Xtr,Xva]); yfit=pd.concat([ytr,yva])
if chosen_model=='Logistic_Interpretable':
    final=Pipeline([('scale',StandardScaler()),
                    ('model',LogisticRegression(max_iter=4000,class_weight='balanced',
                                                 C=logit_C,solver='lbfgs'))])
else:
    final=RandomForestClassifier(class_weight='balanced_subsample',
                                 random_state=42,n_jobs=1,**rf_best)
final_fit_t0=time.perf_counter()
final.fit(Xfit[chosen_feats],yfit)
final_fit_runtime=time.perf_counter()-final_fit_t0
test_pred=final.predict(Xte[chosen_feats])
pd.DataFrame({'true':yte.values,'pred':test_pred},index=Xte.index).to_csv(RES/'test_predictions_final.csv')
tm=metrics(yte,test_pred)
perclass(yte,test_pred).to_csv(RES/'per_class_metrics.csv',index=False)
cm=confusion_matrix(yte,test_pred,labels=classes)
pd.DataFrame(cm,index=classes,columns=classes).to_csv(RES/'confusion_matrix.csv')
json.dump({'chosen_model':chosen_model,'chosen_candidate':chosen_candidate,'n_features':len(chosen_feats),
           'features':chosen_feats,
           'validation_macro_f1':float(chosen['validation_macro_f1']),
           'validation_balanced_accuracy':float(chosen['validation_balanced_accuracy']),
           **tm,'test_size':len(yte),'final_fit_runtime_sec':float(final_fit_runtime)},
          open(RES/'final_test_metrics.json','w'),indent=2)

# Explicit industrial scenario assumption; does not affect model/feature selection.
cost=[]
for c in ['Stains','Dirtiness']:
    i=classes.index(c); support=int(cm[i,:].sum()); fn=int(support-cm[i,i])
    cost.append({'class':c,'operational_assumption':'For inspection prioritization, a missed defect (False Negative) is treated as costly. This is an explicit scenario assumption, not a learned property of the dataset.','cost_priority_assumption':'high','test_support':support,'recall':float(cm[i,i]/support),'false_negatives':fn,'selection_impact':'none'})
pd.DataFrame(cost).to_csv(RES/'cost_sensitive_classes.csv',index=False)

# Back-fill the actual frozen final-fit runtime in the experiment catalog.
exp = pd.read_csv(RES / 'experiments.csv')
exp.loc[exp['Experiment ID'] == 'EXP_07_FINAL_FROZEN_PIPELINE', 'Runtime'] = float(final_fit_runtime)
exp.to_csv(RES / 'experiments.csv', index=False)

# Class-overlap analysis based on final Test confusion; feature separation is estimated from Train only.
pairs=[]
for i,a in enumerate(classes):
    for j,b in enumerate(classes):
        if i<j: pairs.append((int(cm[i,j]+cm[j,i]),a,b,int(cm[i,j]),int(cm[j,i])))
rows=[]
misclass_notes=[]
for total,a,b,ab,ba in sorted(pairs,reverse=True)[:2]:
    xa=Xtr.loc[ytr==a,FEATURES]; xb=Xtr.loc[ytr==b,FEATURES]
    pooled=np.sqrt((xa.var(ddof=1)+xb.var(ddof=1))/2).replace(0,np.nan)
    effect=((xa.mean()-xb.mean()).abs()/pooled).sort_values(ascending=False)
    pair_mask=((yte==a)&(test_pred==b))|((yte==b)&(test_pred==a))
    pair_idx=Xte.index[pair_mask].tolist()
    sub=Xte.loc[pair_mask,chosen_feats]
    f_mean_str=' | '.join([f'{k}:{v:.3f}' for k,v in sub.mean(numeric_only=True).head(5).items()])
    misclass_notes.append({'pair':f'{a} <-> {b}','n_errors':total,'feature_mean':f_mean_str})
    rows.append({'class_a':a,'class_b':b,'bidirectional_errors':total,'a_as_b':ab,'b_as_a':ba,
                 'top_train_separation_features':'|'.join(effect.dropna().head(5).index),
                 'test_misclassified_examples_in_pair':len(pair_idx),
                 'test_example_indices':'|'.join(map(str,pair_idx[:20])),
                 'analysis_note':'Pair ranked by bidirectional final-Test confusion; feature separation is descriptive and computed from Train distributions.'})
pd.DataFrame(rows).to_csv(RES/'class_overlap_analysis.csv',index=False)
pd.DataFrame(misclass_notes).to_csv(RES/'misclassification_analysis.csv',index=False)

# Final-model noise sensitivity: fit the frozen final model on Train only and perturb Validation only.
continuous_selected=[f for f in chosen_feats if Xtr[f].nunique()>2 and not set(Xtr[f].dropna().unique()).issubset({0,1})]
noise_std=Xtr[continuous_selected].std(ddof=0).replace(0,1.0)
noise_model = (Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=4000,class_weight='balanced',C=logit_C,solver='lbfgs'))])
               if chosen_model=='Logistic_Interpretable' else
               RandomForestClassifier(class_weight='balanced_subsample',random_state=42,n_jobs=1,**rf_best))
noise_model.fit(Xtr[chosen_feats],ytr)
rng=np.random.default_rng(2026)
noise_rows=[]
for level in [0.0,0.01,0.05,0.10]:
    Xn=Xva[chosen_feats].astype(float).copy()
    if level:
        eps=rng.normal(0,level,size=(len(Xn),len(continuous_selected)))*noise_std.to_numpy()
        Xn.loc[:,continuous_selected]=Xn[continuous_selected].to_numpy()+eps
    pr=noise_model.predict(Xn)
    mm=metrics(yva,pr); pc=perclass(yva,pr)
    row={'model':chosen_model,'candidate':chosen_candidate,'feature_subset':chosen_candidate,'feature_count':len(chosen_feats),'n_features':len(chosen_feats),
         'evaluation_split':'Validation','noise_level':level,
         'noise_definition':'Gaussian perturbation on continuous selected features; sigma = noise_level * Train feature std; binary features unchanged',**mm}
    for c in classes:
        row[f'precision_{c}']=float(pc.loc[pc['class']==c,'precision'].iloc[0])
        row[f'recall_{c}']=float(pc.loc[pc['class']==c,'recall'].iloc[0])
        row[f'f1_{c}']=float(pc.loc[pc['class']==c,'f1'].iloc[0])
    noise_rows.append(row)
pd.DataFrame(noise_rows).to_csv(RES/'noise_sensitivity.csv',index=False)
(RES/'noise_sensitivity_method.md').write_text(
    f"# Stage 3 noise sensitivity\n\nModel: {chosen_model}\nFeature subset: {chosen_candidate}\n"
    "Evaluation split: Validation\nModel fitting split: Train only\nTest used in noise experiment: No\n"
    "Noise: Gaussian perturbation on continuous selected features; sigma = level * Train feature std; binary features unchanged\n"
    "Levels: 0, 0.01, 0.05, 0.10\n"
)

# Stage-3 diagnostic figures.
import matplotlib.pyplot as plt
cm_df=pd.read_csv(RES/'confusion_matrix.csv',index_col=0)
plt.figure(figsize=(9,7)); plt.imshow(cm_df.values, aspect='auto'); plt.colorbar(label='Count')
plt.xticks(range(len(classes)),classes,rotation=45,ha='right'); plt.yticks(range(len(classes)),classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j,i,str(int(cm_df.iloc[i,j])),ha='center',va='center')
plt.xlabel('Predicted class'); plt.ylabel('True class'); plt.title('Final Test Confusion Matrix'); plt.tight_layout(); plt.savefig(FIG/'confusion_matrix.png',dpi=180); plt.close()

ga_conv=pd.read_csv(RES/'ga_convergence.csv')
plt.figure(figsize=(9,6))
for seed,g in ga_conv.groupby('seed'):
    plt.plot(g['generation'],g['best_fitness'],marker='o',label=f'Seed {int(seed)}')
plt.xlabel('Generation'); plt.ylabel('Best fitness'); plt.title('GA Convergence Across Seeds'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'ga_convergence.png',dpi=180); plt.close()

ga_freq=pd.read_csv(RES/'ga_feature_frequency.csv').sort_values('selection_frequency')
plt.figure(figsize=(10,8)); plt.barh(ga_freq['feature'],ga_freq['selection_frequency']); plt.xlabel('Selection frequency across 3 seeds'); plt.title('GA Feature Selection Frequency'); plt.tight_layout(); plt.savefig(FIG/'ga_feature_frequency.png',dpi=180); plt.close()

leakage = f"""# Leakage Audit

- Raw data files were not modified.
- Stratified Train/Validation/Test indices were fixed before Stage 3 modeling.
- Test rows were excluded from baseline, robustness, noise, filter, GA, hyperparameter tuning, and candidate selection.
- Scaling was inside Pipelines.
- Winsorization learned its percentiles from training data inside a Pipeline.
- Filter selection was computed from X_train only.
- GA fitness used only 3-fold CV on Train.
- Hyperparameter search used Train-only 5-fold CV with seed 2026.
- Unified model/feature candidate selection used the held-out Validation split after Train-only feature/model preparation.
- Noise sensitivity fits the final model on Train only and evaluates perturbed Validation rows; Test is excluded.
- Final Test was evaluated once after fitting the selected pipeline on Train+Validation.
- Outliers were not automatically deleted.

Final selected model: **{chosen_model}**
Final selected candidate: **{chosen_candidate}**
Feature count: **{len(chosen_feats)}**
"""

import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

(RES/'leakage_audit.md').write_text(leakage)
json.dump({'train':len(Xtr),'validation':len(Xva),'test':len(Xte),'chosen_model':chosen_model,'chosen_candidate':chosen_candidate,'chosen_n_features':len(chosen_feats),'chosen_features':chosen_feats},open(RES/'stage3_summary.json','w'),indent=2)
print('Stage 3 full execution completed.')
print('CHOSEN:', chosen_model, chosen_candidate, len(chosen_feats), 'Validation Macro F1=', float(chosen['validation_macro_f1']))
print('TEST:', tm)
