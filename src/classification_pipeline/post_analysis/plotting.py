
# ---- DAVID Enrichment Bar Plot (Fig 7) ----
def plot_david_enrichment(david_tsv, output_path, top_n=10, wrap_width=40, title="Top DAVID Enrichment Terms"):
    """
    Plot Fig 7: horizontal bar chart of top DAVID terms by -log10(FDR).

    Inputs:
        david_tsv: Path to DAVID-exported TSV with columns including 'Term' and 'FDR'
        output_path: Where to save the figure
        top_n: Number of top terms to plot (smallest FDR)
        wrap_width: Maximum characters before wrapping term names
        title: Plot title
    """
    # Load full DAVID results
    df = pd.read_csv(david_tsv, sep="\t")
    # Filter significant terms
    df = df[df["FDR"] < 0.05].copy()
    # Select top_n by smallest FDR
    top = df.nsmallest(top_n, "FDR")
    # Compute -log10(FDR)
    top["neg_log_fdr"] = -np.log10(top["FDR"])
    # Wrap term names
    top["wrapped_term"] = top["Term"].apply(lambda t: textwrap.fill(t, wrap_width))
    # Sort for plotting
    top = top.sort_values("neg_log_fdr", ascending=True)

    plt.figure(figsize=(8, top.shape[0] * 0.5 + 2))
    plt.barh(top["wrapped_term"], top["neg_log_fdr"], align="center")
    plt.xlabel(r"$-\log_{10}(\mathrm{FDR})$")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅  Saved Fig 7 ➜ {output_path}")
import os
import re
import glob
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, recall_score
)
from sklearn.manifold import TSNE
# Only import used metrics functions
from statistical_testing import paired_ttest, rm_anova, tukey_hsd

def plot_tsne(features, labels, title, save_path, perplexity=30, random_state=42):
    """
    t-SNE Plot: visualize high-dimensional sample embeddings in 2D.

    Inputs:
        features: numpy array or DataFrame of shape (n_samples, n_features)
        labels: list or array of class labels (same length as features)
        title: title for the plot
        save_path: where to save the resulting image
        perplexity: t-SNE perplexity (default = 30)
        random_state: seed for reproducibility
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    if hasattr(scatter, 'legend_elements'):
        plt.legend(*scatter.legend_elements(), title="Classes")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_boxplot(df, metric_col, group_col, title, save_path):
    """
    Boxplot: shows median, IQR, and outliers.
    Use when comparing distribution of metric (e.g., F1, AUC) across model groups or thresholds.

    Inputs:
        df: DataFrame with performance metrics per fold
        metric_col: column name of the metric to plot (e.g., "F1")
        group_col: column to group by (e.g., "Model", "Threshold")
        title: plot title
        save_path: path to save .png/.pdf image
    """

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=group_col, y=metric_col, data=df)
    plt.title(title)
    plt.ylabel(metric_col)
    plt.xlabel(group_col)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_violinplot(df, metric_col, group_col, title, save_path):
    """
    Violin plot: shows distribution shape + boxplot summary.
    Use to detect multimodal/skewed distributions across folds.

    Inputs:
        df: DataFrame with fold-wise metrics
        metric_col: metric to plot (e.g., "AUC", "F1")
        group_col: group/category (e.g., "Model")
        title: title string
        save_path: path to save figure
    """

    plt.figure(figsize=(8, 8))
    sns.violinplot(x=group_col, y=metric_col, data=df, inner="box")
    # wrap long category labels at 10 characters
    ax = plt.gca()
    wrapped = [textwrap.fill(lbl.get_text(), width=10) for lbl in ax.get_xticklabels()]
    ax.set_xticklabels(wrapped)
    plt.title(title)
    plt.ylabel(metric_col)
    plt.xlabel(group_col)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_stripplot(df, metric_col, group_col, title, save_path):
    """
    Strip plot: shows individual data points per group.
    Use when you want to highlight every fold's score directly.

    Inputs:
        df: DataFrame with per-fold results
        metric_col: the metric (e.g., "F1", "Recall")
        group_col: group to split by (e.g., "Model" or "Threshold")
        title: title string
        save_path: output filename
    """

    plt.figure(figsize=(8, 6))
    sns.stripplot(x=group_col, y=metric_col, data=df, jitter=True, size=8)
    plt.title(title)
    plt.ylabel(metric_col)
    plt.xlabel(group_col)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_barplot(df, metric_col, group_col, title, save_path):
    """
    Bar plot with error bars: shows mean ± std per group.
    Use for clean summaries of metric performance per model or modality.

    Inputs:
        df: DataFrame with metric values
        metric_col: column with numeric metric
        group_col: category to compare (e.g., "Model", "Threshold")
        title: plot title
        save_path: output image path
    """

    plt.figure(figsize=(8, 6))
    sns.barplot(x=group_col, y=metric_col, data=df, ci="sd")
    plt.title(title)
    plt.ylabel(metric_col)
    plt.xlabel(group_col)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metric_heatmap(df, index_col, column_col, value_col, title, save_path):
    """
    Heatmap: matrix view of scores (e.g., F1 per model × threshold).
    Use to spot high/low performance across combinations.

    Inputs:
        df: DataFrame with one row per (index, column) pair
        index_col: y-axis grouping (e.g., "Model")
        column_col: x-axis grouping (e.g., "Threshold")
        value_col: numeric value to display (e.g., "F1_mean")
        title: plot title
        save_path: path to save figure
    """

    pivot_df = df.pivot(index=index_col, columns=column_col, values=value_col)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
    plt.title("Top Consistent Features (≥2 folds)")
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
                raise KeyError(f"CSV must contain 'y_true' and 'y_score' columns")
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

# ---- Mean ROC by Thresholds ----

# ----------------------------------------------------------------------
# Overlay ROC curves for multiple model / threshold folders
# ----------------------------------------------------------------------
def overlay_roc_curves(model_dirs, output_path, labels=None, title="ROC Curves by Threshold"):
    """
    model_dirs : list[str]
        Each directory must contain *five* CSVs (one per outer fold) with
        columns  'y_true'  and  either  'y_score'  or  'y_proba'.

    labels : list of str, optional
        Custom legend labels for each model_dir; defaults to basenames.

    output_path : str
        Where to save the combined ROC figure.

    The function computes a mean ROC (via interpolation on a common FPR grid)
    for each directory and overlays them in a single plot.
    """
    import glob, os
    base_fpr = np.linspace(0.0, 1.0, 201)       # fine 0–1 grid
    plt.figure(figsize=(7, 7))

    for idx, md in enumerate(model_dirs):
        # Use manual label if provided, otherwise folder basename
        if labels and idx < len(labels):
            label = labels[idx]
        else:
            label = os.path.basename(md.rstrip("/"))
        # gather per‑fold files
        fold_csvs = glob.glob(os.path.join(md, "*.csv"))
        if not fold_csvs:
            print(f"⚠️  No ROC CSVs found in {md}; skipping.")
            continue

        tprs, aucs = [], []
        for csv in fold_csvs:
            df = pd.read_csv(csv)
            y_true  = df["y_true"].values
            if "y_score" in df.columns:
                y_score = df["y_score"].values
            elif "y_proba" in df.columns:
                y_score = df["y_proba"].values
            else:
                raise KeyError(f"{csv} missing 'y_score' or 'y_proba' column")

            fpr, tpr, _ = roc_curve(y_true, y_score)
            interp_tpr = np.interp(base_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_score(y_true, y_score))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)
        plt.plot(base_fpr, mean_tpr,
                 label=f"{label}  (AUC = {mean_auc:.3f} ± {std_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅  Combined ROC figure saved ➜  {output_path}")

# ---- Mean PR Curve Plotting ----

def plot_mean_pr_curves(pr_folder, output_path):
    """
    Group PR CSVs in pr_folder by config prefix, compute mean precision-recall
    curves as true step functions (no linear interpolation), with shaded ±1 std,
    and save the plot to output_path.
    """
    # 1) Collect files by config
    pr_files = glob.glob(os.path.join(pr_folder, "*.csv"))
    config_dict = {}
    for fpath in pr_files:
        name = os.path.basename(fpath).split('_')
        # prune to model_modality if possible
        if len(name) >= 4:
            config = f"{name[2]}_{name[3]}"
        else:
            config = os.path.splitext(os.path.basename(fpath))[0]
        config_dict.setdefault(config, []).append(fpath)

    # 2) Define common recall grid
    base_recall = np.linspace(0.0, 1.0, 101)

    plt.figure(figsize=(8, 8))

    # 3) For each config, build staircase curves and average
    for config, files in config_dict.items():
        fold_aps = []
        fold_step_curves = []  # each is length 101

        for fpath in files:
            df = pd.read_csv(fpath)
            y_true  = df['y_true'].values
            # handle either column name
            y_score = df['y_score'].values if 'y_score' in df.columns else df['y_proba'].values

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            # pad endpoints so recall goes [0,...,1]
            precision = np.concatenate(([1.0], precision, [0.0]))
            recall    = np.concatenate(([0.0], recall,    [1.0]))

            fold_aps.append(average_precision_score(y_true, y_score))

            # build a true step function: for each r in base_recall, find first recall >= r
            step_vals = []
            for r in base_recall:
                idx = np.searchsorted(recall, r, side='right')
                # if we're beyond last point, take the final precision
                if idx >= len(precision):
                    step_vals.append(precision[-1])
                else:
                    step_vals.append(precision[idx])
            fold_step_curves.append(step_vals)

        # convert to array: (n_folds, 101)
        curves = np.array(fold_step_curves)
        mean_prec = curves.mean(axis=0)
        std_prec  = curves.std(axis=0)
        mean_ap   = np.mean(fold_aps)
        std_ap    = np.std(fold_aps)

        # 4) Plot mean step curve + ribbon
        plt.step(base_recall, mean_prec, where='post',
                 label=f"{config} (AP = {mean_ap:.3f} ± {std_ap:.3f})")
        lower = np.clip(mean_prec - std_prec, 0.0, 1.0)
        upper = np.clip(mean_prec + std_prec, 0.0, 1.0)
        plt.fill_between(base_recall, lower, upper, step='post', alpha=0.2)

    # 5) Final styling
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Mean Precision–Recall Curves by Configuration")
    plt.legend(loc="lower left")
    plt.tight_layout()

    # 6) Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_radar_chart(df, model_col, metric_cols, title, save_path):
    """
    Radar/Spider Chart: visualize multiple models across several normalized metrics.
    Use when comparing multi-metric performance in a compact, visual form.

    Inputs:
        df: DataFrame where each row is a model and columns include metrics
        model_col: name of the column with model names
        metric_cols: list of metric column names (e.g., ["AUC", "AUPR", "ACC", "PRE", "REC", "F1"])
        title: plot title
        save_path: where to save the resulting figure
    """
    # Normalize metric columns to [0, 1]
    norm_df = df.copy()
    norm_df[metric_cols] = (df[metric_cols] - df[metric_cols].min()) / (df[metric_cols].max() - df[metric_cols].min())

    labels = metric_cols
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, row in norm_df.iterrows():
        values = row[metric_cols].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row[model_col], marker='o')
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title(title, y=1.1)
    ax.grid(True)
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, 0.1))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




if __name__ == "__main__":

    # Example usage for Fig 7:
    david_tsv    = "/Users/nickq/Documents/Pioneer Academics/Research_Project/src/classification_pipeline/post_analysis/david_res_filtered"   # must have 'Term' and 'FDR' columns
    fig7_output  = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/figures/asset_7/top_enriched_terms"
    plot_david_enrichment(david_tsv, fig7_output, top_n=11)