import os
import glob
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, accuracy_score, recall_score,
    confusion_matrix, f1_score, roc_auc_score
)

def plot_feature_importance_bar(csv_path, output_path):
    """
    Load CSV with 'MeanImportance' and 'StdImportance', sort by mean importance,
    and plot horizontal bar chart with error bars, saving to output_path.
    """
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values("MeanImportance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(df_sorted['Feature'], df_sorted['MeanImportance'], xerr=df_sorted['StdImportance'], color='skyblue', ecolor='black', capsize=3)
    plt.xlabel('Mean Importance')
    plt.title('Feature Importances with Standard Deviation')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_top20_consistent_features(csv_path, output_path):
    """
    Load CSV with 'Raw_Feature_Importances' column containing dict strings,
    parse them, build DataFrame, filter features present in at least 2 folds,
    compute relative importance, plot top 20 features as horizontal bar chart,
    and save to output_path.
    """
    df = pd.read_csv(csv_path)
    raw_dicts = df['Raw_Feature_Importances'].apply(lambda s: ast.literal_eval(s)['Importance_score'])
    imp_df = pd.DataFrame(raw_dicts.tolist())
    valid = imp_df.count(axis=0) >= 2
    filtered = imp_df.loc[:, valid]
    avg_imp = filtered.mean(axis=0).sort_values(ascending=False)
    rel_imp = avg_imp / avg_imp.sum() * 100
    top20 = rel_imp.head(20)

    plt.figure(figsize=(8, 10))
    bars = plt.barh(top20.index, top20.values, color='steelblue')
    plt.xlabel("Relative Importance (%)")
    plt.title("Top 20 Consistent Features (≥2 folds)")
    plt.gca().invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                 f"{width:.1f}%", va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def load_roc_data_by_config(folder):
    """
    Load ROC CSV files from folder, group by config prefix (filename before first '_'),
    and return dict mapping config -> list of DataFrames.
    """
    files = glob.glob(os.path.join(folder, "*.csv"))
    config_dict = {}
    for f in files:
        base = os.path.basename(f)
        config = base.split('_')[0]
        df = pd.read_csv(f)
        if config not in config_dict:
            config_dict[config] = []
        config_dict[config].append(df)
    return config_dict

def plot_mean_roc_curves(roc_folder, output_path):
    """
    For each config group in roc_folder, compute mean ROC curve and plot all on one figure,
    saving to output_path.
    """
    config_dict = load_roc_data_by_config(roc_folder)
    plt.figure(figsize=(8, 8))

    for config, dfs in config_dict.items():
        # Collect all (fpr, tpr) pairs for each fold
        interp_tprs = []
        base_fpr = np.linspace(0, 1, 101)
        aucs = []
        for df in dfs:
            # Always compute ROC from raw labels and scores
            y_true = df['y_true'].values
            if 'y_score' in df.columns:
                y_score = df['y_score'].values
            elif 'y_proba' in df.columns:
                y_score = df['y_proba'].values
            else:
                raise KeyError(f"CSV {f} must contain 'y_true' and 'y_score' columns")
            fpr, tpr, _ = roc_curve(y_true, y_score)
            # interpolate to common FPR grid
            interp_tpr = np.interp(base_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
            # use sklearn's roc_auc_score for exact AUC
            aucs.append(roc_auc_score(y_true, y_score))
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        # average the per-fold AUCs
        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)

        plt.plot(base_fpr, mean_tpr, label=f"{config} (AUC = {mean_auc:.3f} ± {std_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curves by Configuration')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    plot_mean_roc_curves("/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/ablation_experiments/roc_curves_types_gcn",
                         "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/figures/roc_curves_types_gcn.png")