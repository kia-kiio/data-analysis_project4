
import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/processed_dataset.csv'
RES=ROOT/'results'; FIG=ROOT/'figures'
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
RANDOM_STATE=42
TARGET='Class'
TARGET_COLS=['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']
FEATURES=['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity','Maximum_of_Luminosity','Length_of_Conveyer','TypeOfSteel_A300','TypeOfSteel_A400','Steel_Plate_Thickness','Edges_Index','Empty_Index','Square_Index','Outside_X_Index','Edges_X_Index','Edges_Y_Index','Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index','Orientation_Index','Luminosity_Index','SigmoidOfAreas']

# Load processed data produced by stage 1
df=pd.read_csv(DATA)
assert list(df[FEATURES].columns)==FEATURES and TARGET in df.columns
X=df[FEATURES].copy(); y=df[TARGET].copy()

# Stratified split: 70/15/15
X_train, X_tmp, y_train, y_tmp, idx_train, idx_tmp = train_test_split(X,y,df.index,test_size=0.30,stratify=y,random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(X_tmp,y_tmp,idx_tmp,test_size=0.50,stratify=y_tmp,random_state=RANDOM_STATE)
parts=[]
for name, yy in [('Train',y_train),('Validation',y_val),('Test',y_test)]:
 c=yy.value_counts().reindex(TARGET_COLS,fill_value=0)
 for cls,n in c.items(): parts.append({'split':name,'class':cls,'count':int(n),'percentage':float(n/len(yy)*100)})
pd.DataFrame(parts).to_csv(RES/'split_distribution.csv',index=False)

# Descriptive stats on TRAIN for seven representative features
rep=['X_Minimum','Y_Minimum','Pixels_Areas','Sum_of_Luminosity','Steel_Plate_Thickness','Edges_Index','Orientation_Index']
st=[]
for f in rep:
 s=X_train[f]
 st.append({'feature':f,'min':s.min(),'max':s.max(),'mean':s.mean(),'median':s.median(),'std':s.std(),'q25':s.quantile(.25),'q50':s.quantile(.50),'q75':s.quantile(.75),'q90':s.quantile(.90),'q95':s.quantile(.95)})
pd.DataFrame(st).to_csv(RES/'eda_summary.csv',index=False)

# Correlation / redundancy on TRAIN only
corr=X_train.corr(numeric_only=True)
corr.to_csv(RES/'correlation_matrix.csv')
rows=[]
for i,a in enumerate(FEATURES):
 for b in FEATURES[i+1:]:
  r=float(corr.loc[a,b])
  if abs(r)>=0.80:
   rows.append({'feature_1':a,'feature_2':b,'correlation':r,'abs_correlation':abs(r),'redundancy_flag':'high' if abs(r)>=.90 else 'moderate-high'})
red=pd.DataFrame(rows).sort_values('abs_correlation',ascending=False)
red.to_csv(RES/'feature_redundancy.csv',index=False)

# Steel type relationship
steel=pd.DataFrame({
 'metric':['A300_ones','A400_ones','both_one_rows','both_zero_rows','mutually_exclusive'],
 'value':[int((X_train['TypeOfSteel_A300']==1).sum()),int((X_train['TypeOfSteel_A400']==1).sum()),int(((X_train['TypeOfSteel_A300']==1)&(X_train['TypeOfSteel_A400']==1)).sum()),int(((X_train['TypeOfSteel_A300']==0)&(X_train['TypeOfSteel_A400']==0)).sum()),bool(((X_train['TypeOfSteel_A300']+X_train['TypeOfSteel_A400'])<=1).all())]
})
steel_extra = pd.DataFrame({
 'metric':['A300_zeros','A400_zeros','both_one_rows','both_zero_rows','sum_equals_one_rows'],
 'value':[
   int((X_train['TypeOfSteel_A300']==0).sum()),
   int((X_train['TypeOfSteel_A400']==0).sum()),
   int(((X_train['TypeOfSteel_A300']==1)&(X_train['TypeOfSteel_A400']==1)).sum()),
   int(((X_train['TypeOfSteel_A300']==0)&(X_train['TypeOfSteel_A400']==0)).sum()),
   int(((X_train['TypeOfSteel_A300']+X_train['TypeOfSteel_A400'])==1).sum())
 ]
})
steel = pd.concat([steel, steel_extra], ignore_index=True)
steel.to_csv(RES/'steel_type_analysis.csv',index=False)

# Geometric consistency checks on TRAIN only. These are diagnostics, not deletion rules.
geom_rules = [
 ('X_Minimum_le_X_Maximum', (X_train['X_Minimum'] <= X_train['X_Maximum'])),
 ('Y_Minimum_le_Y_Maximum', (X_train['Y_Minimum'] <= X_train['Y_Maximum'])),
 ('Pixels_Areas_positive', (X_train['Pixels_Areas'] > 0)),
 ('X_Perimeter_positive', (X_train['X_Perimeter'] > 0)),
 ('Y_Perimeter_positive', (X_train['Y_Perimeter'] > 0)),
 ('Steel_Plate_Thickness_positive', (X_train['Steel_Plate_Thickness'] > 0)),
 ('Edges_Index_in_[0,1]', X_train['Edges_Index'].between(0,1)),
 ('Empty_Index_in_[0,1]', X_train['Empty_Index'].between(0,1)),
 ('Square_Index_in_[0,1]', X_train['Square_Index'].between(0,1)),
 ('Outside_X_Index_in_[0,1]', X_train['Outside_X_Index'].between(0,1)),
 ('Edges_X_Index_in_[0,1]', X_train['Edges_X_Index'].between(0,1)),
 ('Edges_Y_Index_in_[0,1]', X_train['Edges_Y_Index'].between(0,1)),
 ('Outside_Global_Index_in_[0,1]', X_train['Outside_Global_Index'].between(0,1)),
 ('SigmoidOfAreas_in_[0,1]', X_train['SigmoidOfAreas'].between(0,1)),
]
grows=[]
for rule, mask in geom_rules:
    grows.append({'rule':rule,'valid_count':int(mask.sum()),'invalid_count':int((~mask).sum()),'invalid_percentage':float((~mask).mean()*100)})
pd.DataFrame(grows).to_csv(RES/'geometry_quality.csv',index=False)

# Save exact row indices of the fixed stratified split for downstream reproducibility.
split_idx = pd.DataFrame({
 'row_index': list(idx_train)+list(idx_val)+list(idx_test),
 'split': ['Train']*len(idx_train)+['Validation']*len(idx_val)+['Test']*len(idx_test)
}).sort_values('row_index')
split_idx.to_csv(RES/'split_indices.csv',index=False)

# Scale analysis on TRAIN
scale=[]
for f in FEATURES:
 s=X_train[f]
 scale.append({'feature':f,'min':s.min(),'max':s.max(),'range':s.max()-s.min(),'mean':s.mean(),'std':s.std(),'skewness':s.skew(),'unique_count':s.nunique()})
pd.DataFrame(scale).sort_values('range',ascending=False).to_csv(RES/'scale_analysis.csv',index=False)

# Global and per-class outlier analysis (diagnostic only), thresholds learned from TRAIN
q1=X_train.quantile(.25); q3=X_train.quantile(.75); iqr=q3-q1
lo=q1-1.5*iqr; hi=q3+1.5*iqr
flag_iqr_global=(X_train.lt(lo)|X_train.gt(hi))
row_iqr_global=flag_iqr_global.any(axis=1)

# Global Isolation Forest on robustly scaled training data
Xr=(X_train-X_train.median())/(iqr.replace(0,np.nan)).fillna(1)
iso=IsolationForest(n_estimators=300,contamination=0.05,random_state=RANDOM_STATE,n_jobs=-1)
iso_pred=iso.fit_predict(Xr.fillna(0))
row_iso_global=iso_pred==-1

out_rows=[]
# Global summaries
out_rows += [
 {'scope':'global_train','method':'IQR','class':'ALL','flagged_count':int(row_iqr_global.sum()),'flagged_percentage':float(row_iqr_global.mean()*100),'feature_flag_count':int(flag_iqr_global.sum().sum())},
 {'scope':'global_train','method':'IsolationForest','class':'ALL','flagged_count':int(row_iso_global.sum()),'flagged_percentage':float(row_iso_global.mean()*100),'feature_flag_count':np.nan},
]
# Per-class thresholds and per-class anomaly detectors
per_class_masks={}
for cls in TARGET_COLS:
    m=(y_train.values==cls)
    Xi=X_train.loc[m]
    q1c=Xi.quantile(.25); q3c=Xi.quantile(.75); iqrc=q3c-q1c
    loc=q1c-1.5*iqrc; hic=q3c+1.5*iqrc
    flags_c=(Xi.lt(loc)|Xi.gt(hic))
    row_iqr_c=flags_c.any(axis=1)
    per_class_masks[cls]=row_iqr_c
    # Per-class Isolation Forest; min class size is sufficient for this diagnostic.
    Xrc=(Xi-Xi.median())/(iqrc.replace(0,np.nan)).fillna(1)
    iso_c=IsolationForest(n_estimators=200,contamination=0.05,random_state=RANDOM_STATE,n_jobs=-1)
    pred_c=iso_c.fit_predict(Xrc.fillna(0))
    row_iso_c=pred_c==-1
    out_rows += [
      {'scope':'within_class_train','method':'IQR','class':cls,'flagged_count':int(row_iqr_c.sum()),'flagged_percentage':float(row_iqr_c.mean()*100),'feature_flag_count':int(flags_c.sum().sum())},
      {'scope':'within_class_train','method':'IsolationForest','class':cls,'flagged_count':int(row_iso_c.sum()),'flagged_percentage':float(row_iso_c.mean()*100),'feature_flag_count':np.nan},
    ]

# Feature-level global IQR counts
for f in FEATURES:
 out_rows.append({'scope':'feature_global_train','method':'IQR','class':'ALL','flagged_count':int(flag_iqr_global[f].sum()),'flagged_percentage':float(flag_iqr_global[f].mean()*100),'feature_flag_count':np.nan,'feature':f})
out=pd.DataFrame(out_rows)
out.to_csv(RES/'outlier_analysis.csv',index=False)

# concise summary for downstream robustness
summary={
 'train_n':len(X_train),'validation_n':len(X_val),'test_n':len(X_test),
 'iqr_global_flagged_rows':int(row_iqr_global.sum()),'iqr_global_flagged_pct':float(row_iqr_global.mean()*100),
 'isolation_global_flagged_rows':int(row_iso_global.sum()),'isolation_global_flagged_pct':float(row_iso_global.mean()*100),
 'both_global_methods_flagged_rows':int((row_iqr_global & row_iso_global).sum()),
 'top_iqr_features':flag_iqr_global.sum().sort_values(ascending=False).head(10).astype(int).to_dict(),
 'high_corr_pairs':int(len(red))
}
(RES/'stage2_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
# Figures
plt.figure(figsize=(11,7)); sns.boxplot(data=pd.DataFrame({'Class':y_train.values,'Pixels_Areas':X_train['Pixels_Areas'].values}),x='Class',y='Pixels_Areas'); plt.xticks(rotation=25); plt.title('Pixels_Areas by Fault Class (Train)'); plt.tight_layout(); plt.savefig(FIG/'boxplot_by_class.png',dpi=180); plt.close()

def scatter(a,b,path):
 plt.figure(figsize=(9,7)); sns.scatterplot(x=X_train[a],y=X_train[b],hue=y_train,palette='tab10',s=28,alpha=.7); plt.title(f'{a} vs {b} by Fault Class (Train)'); plt.xlabel(a); plt.ylabel(b); plt.legend(title='Class',bbox_to_anchor=(1.02,1),loc='upper left'); plt.tight_layout(); plt.savefig(FIG/path,dpi=180); plt.close()
scatter('Pixels_Areas','Sum_of_Luminosity','scatter_plot_1.png')
scatter('X_Minimum','X_Maximum','scatter_plot_2.png')
plt.figure(figsize=(14,11)); sns.heatmap(corr,cmap='coolwarm',center=0,square=False,cbar_kws={'label':'Pearson correlation'}); plt.title('Feature Correlation Matrix (Train)'); plt.tight_layout(); plt.savefig(FIG/'correlation_matrix.png',dpi=180); plt.close()
plot_features=['X_Minimum','Pixels_Areas','Sum_of_Luminosity','Steel_Plate_Thickness','Edges_Index','Square_Index','Luminosity_Index','SigmoidOfAreas']
fig,axes=plt.subplots(4,2,figsize=(13,15)); axes=axes.ravel()
for ax,f in zip(axes,plot_features):
    vals=X_train[f].replace([np.inf,-np.inf],np.nan).dropna()
    ax.hist(vals,bins=35)
    ax.set_title(f)
    ax.set_xlabel('Feature value'); ax.set_ylabel('Frequency')
fig.suptitle('Selected Feature Distributions (Train)',fontsize=15)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(FIG/'feature_distributions.png',dpi=180); plt.close(fig)
# outlier feature bar
feat_counts=flag_iqr_global.sum().sort_values(ascending=False).head(15)
plt.figure(figsize=(10,7)); feat_counts.sort_values().plot(kind='barh'); plt.title('Top Features by Global IQR Flags (Train)'); plt.xlabel('Flagged observations'); plt.tight_layout(); plt.savefig(FIG/'outlier_analysis.png',dpi=180); plt.close()
print(json.dumps(summary,indent=2,ensure_ascii=False))
print('SPLIT',len(X_train),len(X_val),len(X_test))
print('RED',red.head(10).to_string(index=False))
