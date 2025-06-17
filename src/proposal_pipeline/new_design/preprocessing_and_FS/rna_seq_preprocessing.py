#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Paths
INPUT_CSV = "/Users/nickq/Documents/Pioneer Academics/Research_Project/proposal_data/datasets_unprocessed/rna_plus_clinical.csv"
OUTPUT_CSV = "/Users/nickq/Documents/Pioneer Academics/Research_Project/proposal_data/datasets_processed/rna_plus_clinical_final.csv"


def main():
    # 1) Read the combined RNA + clinical file
    df = pd.read_csv(INPUT_CSV, dtype={"PATNO": str})
    print(f"[1] Loaded data: {df.shape[0]} samples, {df.shape[1]} columns")

    # 2) Encode SEX as binary flag
    df["SEX_M"] = df["SEX"].astype(int)
    df = df.drop(columns=["SEX"], errors="ignore")

    # 3) Identify column groups
    clinical_cont = ["age_at_visit", "EDUCYRS"]
    cognitive_cont = ["moca", "moca_12m", "moca_change"]
    clinical_bin = ["SEX_M"]
    all_cols = df.columns.tolist()
    # Any column not PATNO or clinical is a gene TPM
    gene_cols = [c for c in all_cols if c not in (["PATNO"] + clinical_cont + clinical_bin + cognitive_cont)]
    print(f"[2] Encoded SEX, clinical & cognitive continuous: {len(clinical_cont + cognitive_cont)}, binary: {len(clinical_bin)}")
    print(f"[3] Identified {len(gene_cols)} gene columns")

    # --- Transcriptomics QC ---
    # 4) Drop genes with >5% missing values
    expr = df[gene_cols]
    keep = expr.columns[expr.isna().mean() <= 0.05]
    df = df[["PATNO"] + clinical_cont + clinical_bin + cognitive_cont + keep.tolist()]
    gene_cols = keep.tolist()
    print(f"[3] QC1 missingness → {len(gene_cols)} genes")

    # 5) Drop genes with TPM <1 in >80% samples
    expr = df[gene_cols]
    keep = expr.columns[(expr >= 1).sum() >= 0.2 * len(df)]
    df = df[["PATNO"] + clinical_cont + clinical_bin + cognitive_cont + keep.tolist()]
    gene_cols = keep.tolist()
    print(f"[4] QC2 low-expression → {len(gene_cols)} genes")

    # 6) Remove one of any gene pair with |r| > 0.9
    corr = df[gene_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > 0.9)]
    keep = [g for g in gene_cols if g not in drop]
    df = df[["PATNO"] + clinical_cont + clinical_bin + cognitive_cont + keep]
    gene_cols = keep
    print(f"[5] QC3 correlation prune → {len(gene_cols)} genes")
    #---


    # Remove intermediate cognitive columns not needed downstream
    df = df.drop(columns=['moca', 'moca_12m'], errors='ignore')
    # 7) Write out the final preprocessed DataFrame
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[6] Written pandas-processed RNA + clinical → {OUTPUT_CSV}")
    print(f"Total gene features retained: {len(gene_cols)}")


if __name__ == "__main__":
    main()