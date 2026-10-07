
import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def save_class_distribution(counts,path):
    ax=counts.plot(kind="bar",figsize=(9,5)); ax.set_title("Steel Plate Fault Class Distribution"); ax.set_xlabel("Class"); ax.set_ylabel("Count"); plt.tight_layout(); plt.savefig(path,dpi=180); plt.close()

def save_feature_distributions(df,features,path):
    fig,axes=plt.subplots(4,2,figsize=(12,14)); axes=axes.ravel()
    for ax,f in zip(axes,features): df[f].hist(ax=ax,bins=30); ax.set_title(f); ax.set_xlabel(f); ax.set_ylabel("Count")
    for ax in axes[len(features):]: ax.axis("off")
    fig.suptitle("Representative Feature Distributions",fontsize=15); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
