if __name__ == "__main__":
    # ------------------------------------------------------------------
    #  Provide the *four* threshold folders here – nothing else to tweak
    # ------------------------------------------------------------------
    model_dirs = [
        "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/thresholds/thres_-2/for_figures",
        "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/thresholds/thres_-3/for_figures",
        "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/thresholds/thres_-4/for_figures",
        "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/thresholds/thres_-5/for_figures"
    ]

    # Custom labels for each threshold
    labels = ["Threshold -2", "Threshold -3", "Threshold -4", "Threshold -5"]

    overlay_roc_curves(
        model_dirs,
        "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/figures/asset_1/figures/combined_roc.png",
        labels=labels
    )

    # ------------------------------------------------------------------
    # 4) Compute per‑fold AUCs, run ANOVA + Tukey, and save Table 1
    # ------------------------------------------------------------------

    metrics_records = []
    fold_pattern = re.compile(r"fold[_]?(\d+)", re.IGNORECASE)

    for root, lab in zip(model_dirs, labels):
        csv_files = glob.glob(os.path.join(root, "*.csv"))
        if not csv_files:
            print(f"⚠️  No CSVs found in {root}; skipping metrics extraction.")
            continue
        for csv_path in csv_files:
            df_tmp = pd.read_csv(csv_path)
            y_true = df_tmp["y_true"].values
            if "y_score" in df_tmp.columns:
                y_score = df_tmp["y_score"].values
            elif "y_proba" in df_tmp.columns:
                y_score = df_tmp["y_proba"].values
            else:
                raise KeyError(f"{csv_path} missing y_score/y_proba column")

            # Infer fold number from filename
            m = fold_pattern.search(os.path.basename(csv_path))
            fold_id = int(m.group(1)) if m else -1

            y_pred = (y_score >= 0.5).astype(int)
            metrics_records.append({
                "threshold": lab,
                "fold": fold_id,
                "AUC": roc_auc_score(y_true, y_score),
                "F1": f1_score(y_true, y_pred),
                "Recall": recall_score(y_true, y_pred)
            })

    metrics_df = pd.DataFrame(metrics_records)

    # ----------------- Stats: RM‑ANOVA & Tukey for each metric -----------------
    tables_dir = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/figures/asset_1/tables"
    os.makedirs(tables_dir, exist_ok=True)
    table1_path = os.path.join(tables_dir, "Table2_threshold_metrics.csv")
    stats_path  = os.path.join(tables_dir, "Table2_metrics_anova_tukey.txt")

    with open(stats_path, "w") as fh:
        for metric in ["AUC", "F1", "Recall"]:
            fh.write(f"Repeated-measures ANOVA on {metric} vs Threshold\n")
            aov_table = rm_anova(data=metrics_df, dv=metric, within="threshold", subject="fold")
            fh.write(aov_table.to_string() + "\n\n")
            tuk = tukey_hsd(metrics_df[metric].values, metrics_df["threshold"].astype(str).values)
            fh.write(f"Tukey HSD results ({metric}):\n")
            fh.write(tuk.summary().as_text() + "\n\n")

    # ----------------- Aggregate & Save Table -----------------
    table1 = metrics_df.groupby("threshold").agg(
        AUC_mean=("AUC","mean"),    AUC_std=("AUC","std"),
        F1_mean=("F1","mean"),      F1_std=("F1","std"),
        Recall_mean=("Recall","mean"), Recall_std=("Recall","std")
    ).reset_index()
    table1.to_csv(table1_path, index=False)
    print(f"✅  Saved full metrics Table ➜ {table1_path}")
    print(f"✅  Saved ANOVA/Tukey stats ➜ {stats_path}")
