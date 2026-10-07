
import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pathlib import Path
import pandas as pd, numpy as np, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'
classes=['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']
features=pd.read_csv(RES/'data_dictionary.csv').query("role=='feature'")['column'].tolist()
df=pd.read_csv(ROOT/'data/processed/processed_dataset.csv')
s=pd.read_csv(RES/'split_indices.csv')
tr=s.loc[s.split=='Train','row_index'].astype(int); va=s.loc[s.split=='Validation','row_index'].astype(int)
Xtr=df.loc[tr,features]; ytr=df.loc[tr,'Class']; Xva=df.loc[va,features]; yva=df.loc[va,'Class']
rows=[]
for name,cw in [('None',None),('balanced','balanced'),('balanced_subsample','balanced_subsample')]:
    t=time.time(); m=RandomForestClassifier(n_estimators=80,random_state=42,n_jobs=1,class_weight=cw); m.fit(Xtr,ytr); p=m.predict(Xva)
    r={'Model':'RandomForest_Nonlinear','Class_Weight':name,'Accuracy':accuracy_score(yva,p),'Balanced_Accuracy':balanced_accuracy_score(yva,p),'Macro_F1':f1_score(yva,p,average='macro'),'Weighted_F1':f1_score(yva,p,average='weighted')}
    rec=recall_score(yva,p,labels=classes,average=None,zero_division=0)
    for c,v in zip(classes,rec): r['Recall_'+c]=v
    r['Runtime_sec']=time.time()-t
    rows.append(r)
pd.DataFrame(rows).to_csv(RES/'class_weight_comparison.csv',index=False)
# Final reciprocal-error analysis is produced by stage3_analysis.py from the single frozen Test evaluation.
# This supplementary script intentionally does not access Test.
