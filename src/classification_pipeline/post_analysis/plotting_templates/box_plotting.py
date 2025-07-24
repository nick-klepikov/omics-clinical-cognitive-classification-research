# ------------------------------------------------------------------
#  Provide one folder per *feature size* here
#  (leave the list empty for now; fill paths manually when ready)
# ------------------------------------------------------------------
feature_dirs = [
    "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/feature_dim/thres_-4/40f/metrics_and_features",
    "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/feature_dim/thres_-4/80f/metrics_and_features",
    "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/feature_dim/thres_-4/100f/metrics_and_features",
    "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/feature_dim/thres_-4/120f/metrics_and_features",
    "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/feature_dim/thres_-4/200f/metrics_and_features",
]

# Custom labels for each feature size (must align with feature_dirs order)
feature_labels = ["40 features", "80 features", "100 features", "120 features", "200 features"]

# ------------------------------------------------------------------
# 4) Compute per‑fold AUCs, run ANOVA + Tukey, and save Table 1
# ------------------------------------------------------------------

metrics_records = []
fold_pattern = re.compile(r"fold[_]?(\d+)", re.IGNORECASE)

for root, lab in zip(feature_dirs, feature_labels):
    csv_files = glob.glob(os.path.join(root, "fold_*_selected_metrics.csv"))
    if not csv_files:
        print(f"⚠️  No CSVs found in {root}; skipping metrics extraction.")
        continue
    for csv_path in csv_files:
        df_tmp = pd.read_csv(csv_path)

        # infer fold number
        m = fold_pattern.search(os.path.basename(csv_path))
        fold_id = int(m.group(1)) if m else -1

        # extract precomputed metrics from the summary CSV
        auc = df_tmp["AUC"].iloc[0]
        f1 = df_tmp["F1"].iloc[0]
        recall = df_tmp["Recall"].iloc[0]

        metrics_records.append({
            "Feature Size": lab,
            "fold": fold_id,
            "AUC": auc,
            "F1": f1,
            "Recall": recall
        })

metrics_df = pd.DataFrame(metrics_records)

# Prepare tables output directory for box plots and stats
tables_dir = "/data/figures/asset_2"
os.makedirs(tables_dir, exist_ok=True)

# ------------------------------------------------------------------
#  Box plot: AUC across feature sizes
# ------------------------------------------------------------------
if feature_dirs:  # only run if list is populated
    plot_boxplot(
        metrics_df,
        metric_col="F1",
        group_col="Feature Size",
        title="F1 Distribution Across Feature Counts",
        save_path=os.path.join(tables_dir, "f1_by_feature_size.png")
    )
    print("✅  Saved AUC box plot ➜", os.path.join(tables_dir, "f1_by_feature_size.png"))

# ----------------- Stats: RM‑ANOVA & Tukey for each metric -----------------
table1_path = os.path.join(tables_dir, "Table2_threshold_metrics.csv")
stats_path = os.path.join(tables_dir, "Table2_metrics_anova_tukey.txt")

with open(stats_path, "w") as fh:
    for metric in ["AUC", "F1", "Recall"]:
        fh.write(f"Repeated-measures ANOVA on {metric} vs Threshold\n")
        aov_table = rm_anova(data=metrics_df, dv=metric, within="Feature Size", subject="fold")
        fh.write(aov_table.to_string() + "\n\n")
        tuk = tukey_hsd(metrics_df[metric].values, metrics_df["Feature Size"].astype(str).values)
        fh.write(f"Tukey HSD results ({metric}):\n")
        fh.write(tuk.summary().as_text() + "\n\n")

# ----------------- Aggregate & Save Table -----------------
table1 = metrics_df.groupby("Feature Size").agg(
    AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"),
    F1_mean=("F1", "mean"), F1_std=("F1", "std"),
    Recall_mean=("Recall", "mean"), Recall_std=("Recall", "std")
).reset_index()
table1.to_csv(table1_path, index=False)
print(f"✅  Saved full metrics Table ➜ {table1_path}")
print(f"✅  Saved ANOVA/Tukey stats ➜ {stats_path}")
