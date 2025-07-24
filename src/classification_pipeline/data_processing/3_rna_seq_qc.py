import pandas as pd
import numpy as np
import argparse

def main(threshold):
    for fold in range(5):
        INPUT_CSV = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/final_datasets_unprocessed/rna_plus_clinical_fold_{fold}_thresh_{threshold}.csv"
        OUTPUT_CSV = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/final_datasets_processed/rna_plus_clinical_final_fold_{fold}_thresh_{threshold}.csv"
        df = pd.read_csv(INPUT_CSV, dtype={"PATNO": str})
        clinical_cont = ["age_at_visit", "EDUCYRS"]
        cognitive_cont = ["label"]
        clinical_bin = ["SEX_M"]
        all_cols = df.columns.tolist()
        # Identify gene columns
        gene_cols = [c for c in all_cols if c not in (["PATNO"] + clinical_cont + clinical_bin + cognitive_cont)]
        # Remove genes with >5% NaNs
        expr = df[gene_cols]
        keep = expr.columns[expr.isna().mean() <= 0.05]
        df = df[["PATNO"] + clinical_cont + clinical_bin + cognitive_cont + keep.tolist()]
        gene_cols = keep.tolist()
        # Drop low-expressed genes
        expr = df[gene_cols]
        keep = expr.columns[(expr >= 1).sum() >= 0.2 * len(df)]
        df = df[["PATNO"] + clinical_cont + clinical_bin + cognitive_cont + keep.tolist()]
        gene_cols = keep.tolist()
        # Prune correlated genes
        corr = df[gene_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > 0.9)]
        keep = [g for g in gene_cols if g not in drop]
        df = df[["PATNO"] + clinical_cont + clinical_bin + cognitive_cont + keep]
        # Save processed data
        df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, choices=[-2, -3, -4, -5], required=True, help="Threshold value for input files")
    args = parser.parse_args()
    main(args.threshold)