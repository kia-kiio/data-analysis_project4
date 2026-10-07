
import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
raw=ROOT/'data/raw'
res=ROOT/'results'; fig=ROOT/'figures'
res.mkdir(exist_ok=True); fig.mkdir(exist_ok=True)

names=[x.strip() for x in (raw/'Faults27x7_var').read_text().splitlines() if x.strip()]
df=pd.read_csv(raw/'Faults.NNA', sep='\t', header=None, names=names)
features=names[:27]; targets=names[27:]
assert df.shape[1]==34 and len(features)==27 and len(targets)==7
onehot_sum=df[targets].sum(axis=1)
assert df[targets].isin([0,1]).all().all(), 'Target columns must contain only 0/1 values'
assert (onehot_sum==1).all(), f'Invalid One-Hot rows: {(onehot_sum!=1).sum()}'
assert df[features].isna().sum().sum()==0
assert df[targets].isna().sum().sum()==0
class_map={t:i for i,t in enumerate(targets)}
df['Class']=df[targets].idxmax(axis=1)
df['Class_ID']=df['Class'].map(class_map)

# Processed dataset
proc=df[features+['Class','Class_ID']].copy()
proc.to_csv(ROOT/'data/processed/processed_dataset.csv', index=False)

# target quality: explicit row-level validation
zero_mask=onehot_sum.eq(0)
multi_mask=onehot_sum.gt(1)
rows=[]
for t in targets:
    rows.append({'target':t,'positive_count':int(df[t].sum()),'positive_rate_pct':float(df[t].mean()*100),'unique_values':int(df[t].nunique()),'valid_binary':bool(df[t].isin([0,1]).all())})
rows += [
 {'target':'OneHot_row_sum','positive_count':int(onehot_sum.sum()),'positive_rate_pct':float(onehot_sum.mean()*100),'unique_values':int(onehot_sum.nunique()),'valid_binary':bool((onehot_sum==1).all())},
 {'target':'zero_label_rows','positive_count':int(zero_mask.sum()),'positive_rate_pct':float(zero_mask.mean()*100),'unique_values':int(zero_mask.sum()),'valid_binary':True},
 {'target':'multi_label_rows','positive_count':int(multi_mask.sum()),'positive_rate_pct':float(multi_mask.mean()*100),'unique_values':int(multi_mask.sum()),'valid_binary':True},
 {'target':'invalid_label_rows','positive_count':int((~df[targets].isin([0,1]).all(axis=1)).sum()),'positive_rate_pct':float((~df[targets].isin([0,1]).all(axis=1)).mean()*100),'unique_values':int((~df[targets].isin([0,1]).all(axis=1)).sum()),'valid_binary':True}
]
tq=pd.DataFrame(rows)
tq.to_csv(res/'target_quality_report.csv',index=False)
(res/'target_quality_affected_indices.csv').write_text('row_index	issue\n' + ''.join(f'{i}\t'+('zero_label' if zero_mask.loc[i] else 'multi_label')+'\n' for i in df.index if zero_mask.loc[i] or multi_mask.loc[i]))

# class distribution
cd=df['Class'].value_counts().reindex(targets)
cd_tbl=pd.DataFrame({'class':cd.index,'count':cd.values,'percentage':cd.values/len(df)*100})
cd_tbl.to_csv(res/'class_distribution.csv',index=False)

# Complete data dictionary statistics for all 27 features + 7 targets
# Descriptions are limited to terminology supported by the supplied variable names/project docs.
feature_groups={
 'X_Minimum':'minimum X coordinate of defect bounding region', 'X_Maximum':'maximum X coordinate of defect bounding region',
 'Y_Minimum':'minimum Y coordinate of defect bounding region', 'Y_Maximum':'maximum Y coordinate of defect bounding region',
 'Pixels_Areas':'area measured in pixels', 'X_Perimeter':'X-direction perimeter measure', 'Y_Perimeter':'Y-direction perimeter measure',
 'Sum_of_Luminosity':'sum of luminosity values in the defect region', 'Minimum_of_Luminosity':'minimum luminosity in the defect region',
 'Maximum_of_Luminosity':'maximum luminosity in the defect region', 'Length_of_Conveyer':'conveyor length/position-related measurement',
 'TypeOfSteel_A300':'one-hot indicator for steel type A300', 'TypeOfSteel_A400':'one-hot indicator for steel type A400',
 'Steel_Plate_Thickness':'steel plate thickness', 'Edges_Index':'edge-related index', 'Empty_Index':'empty-region index',
 'Square_Index':'square-shape index', 'Outside_X_Index':'outside X index', 'Edges_X_Index':'X-direction edge index',
 'Edges_Y_Index':'Y-direction edge index', 'Outside_Global_Index':'global outside index', 'LogOfAreas':'log-transformed area index',
 'Log_X_Index':'log-transformed X index', 'Log_Y_Index':'log-transformed Y index', 'Orientation_Index':'orientation index',
 'Luminosity_Index':'luminosity index', 'SigmoidOfAreas':'sigmoid-transformed area index'
}
rows=[]
for c in names:
    s=df[c]
    rows.append({'column':c,'role':'feature' if c in features else 'target_one_hot','dtype':str(s.dtype),
                 'description':feature_groups.get(c,'one-hot fault-class indicator'),
                 'min':float(s.min()),'max':float(s.max()),'mean':float(s.mean()),'median':float(s.median()),'std':float(s.std()),
                 'q25':float(s.quantile(.25)),'q50':float(s.quantile(.50)),'q75':float(s.quantile(.75)),
                 'q90':float(s.quantile(.90)),'q95':float(s.quantile(.95)),'missing_count':int(s.isna().sum()),'unique_count':int(s.nunique())})
pd.DataFrame(rows).to_csv(res/'data_dictionary.csv',index=False)

# Stage-1 full-feature descriptive summary retained separately; Stage 2 will replace eda_summary with the required 7-feature TRAIN-only view.
pd.DataFrame([{'feature':c,'dtype':str(df[c].dtype),'n_unique':int(df[c].nunique()),'missing':int(df[c].isna().sum()),'mean':float(df[c].mean()),'std':float(df[c].std()),'min':float(df[c].min()),'q25':float(df[c].quantile(.25)),'median':float(df[c].median()),'q75':float(df[c].quantile(.75)),'max':float(df[c].max())} for c in features]).to_csv(res/'eda_full_raw_summary.csv',index=False)

# validation JSON
quality={
 'rows':int(len(df)),'feature_count':27,'target_count':7,
 'missing_values':int(df[names].isna().sum().sum()),
 'duplicate_rows':int(df[names].duplicated().sum()),
 'onehot_invalid_rows':int((onehot_sum!=1).sum()),
 'classes':targets,
 'class_counts':{k:int(v) for k,v in cd.items()}
}
(res/'stage1_validation.json').write_text(json.dumps(quality,indent=2))

# class distribution figure
plt.figure(figsize=(9,5)); plt.bar(cd.index,cd.values); plt.xticks(rotation=35,ha='right'); plt.ylabel('Count'); plt.title('Steel Plate Fault Class Distribution'); plt.tight_layout(); plt.savefig(fig/'class_distribution.png',dpi=160); plt.close()

print(json.dumps(quality,indent=2))
