if __name__ == "__main__":
    # ------------------------------------------------------------------
    #  Provide one folder per *graph configuration* here
    #  (leave paths commented; fill when ready)
    # ------------------------------------------------------------------
    config_dirs = [
        "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/single_gcn_model_tweaking/thres_-4/80f/full_graph/metrics_and_features",
        "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/single_gcn_model_tweaking/thres_-4/80f/sparse_graph_10n/metrics_and_features",
        "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/single_gcn_model_tweaking/thres_-4/80f/sparse_graph_50n/metrics_and_features",
        "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/single_gcn_model_tweaking/thres_-4/80f/thres_tuning_sparse_10n/metrics_and_features",
        "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/single_gcn_model_tweaking/thres_-4/80f/thres_tuning_sparse_50n/metrics_and_features",
    ]

    config_labels = [
        "Full graph",
        "Sparse 10n",
        "Sparse 50n",
        "Sparse 10n + thresh tuning",
        "Sparse 50n + thresh tuning",
    ]

    # ------------------------------------------------------------------
    # 4) Compute per‑fold AUCs, run ANOVA + Tukey, and save Table 1
    # ------------------------------------------------------------------

    metrics_records = []
    fold_pattern = re.compile(r"fold[_]?(\d+)", re.IGNORECASE)

    for root, lab in zip(config_dirs, config_labels):
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
            auc    = df_tmp["AUC"].iloc[0]
            f1     = df_tmp["F1"].iloc[0]
            recall = df_tmp["Recall"].iloc[0]
            metrics_records.append({
                "Config": lab,
                "fold": fold_id,
                "AUC": auc,
                "F1": f1,
                "Recall": recall
            })

    metrics_df = pd.DataFrame(metrics_records)

    # Prepare tables output directory for violin plots and stats
    tables_dir = "/data/figures/asset_3"
    os.makedirs(tables_dir, exist_ok=True)

    # ------------------------------------------------------------------
    #  Violin plot: F1 across graph configurations
    # ------------------------------------------------------------------
    if config_dirs:  # only run if list is populated
        plot_violinplot(
            metrics_df,
            metric_col="F1",
            group_col="Config",
            title="F1 Distribution Across Graph Configurations",
            save_path=os.path.join(tables_dir, "f1_by_graph_config.png")
        )
        print("✅  Saved F1 violin plot ➜", os.path.join(tables_dir, "f1_by_graph_config.png"))

    # ----------------- Stats: RM‑ANOVA & Tukey for each metric -----------------
    table1_path = os.path.join(tables_dir, "Table3_graph_config_metrics.csv")
    stats_path  = os.path.join(tables_dir, "Table3_graph_config_anova_tukey.txt")

    with open(stats_path, "w") as fh:
        for metric in ["AUC", "F1", "Recall"]:
            fh.write(f"Repeated-measures ANOVA on {metric} vs Config\n")
            aov_table = rm_anova(data=metrics_df, dv=metric, within="Config", subject="fold")
            fh.write(aov_table.to_string() + "\n\n")
            tuk = tukey_hsd(metrics_df[metric].values, metrics_df["Config"].astype(str).values)
            fh.write(f"Tukey HSD results ({metric}):\n")
            fh.write(tuk.summary().as_text() + "\n\n")

    # ----------------- Aggregate & Save Table -----------------
    table1 = metrics_df.groupby("Config").agg(
        AUC_mean=("AUC","mean"),    AUC_std=("AUC","std"),
        F1_mean=("F1","mean"),      F1_std=("F1","std"),
        Recall_mean=("Recall","mean"), Recall_std=("Recall","std")
    ).reset_index()
    table1.to_csv(table1_path, index=False)
    print(f"✅  Saved full metrics Table ➜ {table1_path}")
    print(f"✅  Saved ANOVA/Tukey stats ➜ {stats_path}")
