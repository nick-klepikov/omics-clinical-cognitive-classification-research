#!/usr/bin/env python3
"""
create_mastertable.py

Merge SNP‐ and RNA‐feature tables (each including clinical + moca_change)
into one mastertable for GCN training.
"""

import os
import pandas as pd


def main():
    # Paths to your per‐modality feature tables:
    snp_csv = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/snp_train_transformed_features.csv"
    rna_csv = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/rna_seq_train_transformed_features.csv"

    # Load them (PATNO as index)
    df_snp = pd.read_csv(snp_csv, dtype={'PATNO': str}).set_index('PATNO')
    df_rna = pd.read_csv(rna_csv, dtype={'PATNO': str}).set_index('PATNO')

    # Sanity check: both must have the same clinical columns+label
    clinical_and_label = ['age_at_visit', 'SEX_M', 'EDUCYRS', 'moca_change']

    # Merge on PATNO (inner join ensures we keep only subjects with both sets)
    df_master = df_snp.join(df_rna.drop(columns=clinical_and_label, errors="ignore"), how='inner')
    # Note: we drop the duplicated clinical+label columns from the RNA side

    # Final check
    print("Mastertable shape:", df_master.shape)
    print("Columns breakdown:")
    print(" • Clinical/label:", clinical_and_label)
    print(" • SNP features:", [c for c in df_snp.columns if c not in clinical_and_label][:5], "...",
          f"({len(df_snp.columns) - len(clinical_and_label)} total)")
    print(" • RNA features:", [c for c in df_rna.columns if c not in clinical_and_label][:5], "...",
          f"({len(df_rna.columns) - len(clinical_and_label)} total)")

    # Save
    out_path = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/fused_datasets/final_mastertable.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_master.to_csv(out_path, index=True)
    print(f"Mastertable written to {out_path}")


if __name__ == "__main__":
    main()