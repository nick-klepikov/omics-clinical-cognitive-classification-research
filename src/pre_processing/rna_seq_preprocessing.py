#!/usr/bin/env python3
"""

NumPy-based RNA-Seq & clinical preprocessing:

"""

import pandas as pd
import numpy as np

# === CONFIGURE PATHS ===
INPUT_CSV  = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_unprocessed/rna_plus_clinical.csv"
OUTPUT_CSV = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/rna_plus_clinical_final.csv"

def main():
    # 1) Load data
    df = pd.read_csv(INPUT_CSV, dtype={"PATNO": str})

    # 2) Drop unused columns
    df = df.drop(columns=[
        "EVENT_ID", "subgroup", "race", "moca", "moca_12m"
    ], errors="ignore")

    # 3) Encode SEX as binary
    df["SEX_M"] = df["SEX"].astype(int)
    df = df.drop(columns=["SEX"], errors="ignore")

    # Define clinical continuous and binary columns
    clin_cont = ["age_at_visit", "EDUCYRS"]
    clin_bin  = ["SEX_M"]

    # 4) Identify gene TPM columns
    exclude = set(["PATNO"]) | set(clin_cont) | set(clin_bin)
    gene_cols = [c for c in df.columns if c not in exclude]

    # 5) Log2-transform TPMs
    df[gene_cols] = np.log2(df[gene_cols] + 1)

    # 6) Build NumPy feature array
    feature_cols = clin_cont + clin_bin + gene_cols
    X = df[feature_cols].to_numpy(dtype=np.float32)

    # 7) Mean-impute and z-score normalize continuous features
    # Compute column means for continuous and binary features
    col_means = np.nanmean(X, axis=0)
    # Replace NaNs with column means
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    # Compute means and stds
    col_mu = X.mean(axis=0)
    col_sigma = X.std(axis=0)
    col_sigma[col_sigma == 0] = 1.0
    # Z-score normalize all columns, then restore binary flags
    X = (X - col_mu) / col_sigma
    # Ensure binary columns are strictly 0 or 1
    for i, col in enumerate(feature_cols):
        if col in clin_bin:
            X[:, i] = (X[:, i] > 0).astype(np.float32)

    # 8) Rebuild DataFrame and save
    df_out = pd.DataFrame(X, columns=feature_cols, index=df.index)
    df_out.insert(0, "PATNO", df["PATNO"])
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote NumPy-processed RNA-Seq + clinical data → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()