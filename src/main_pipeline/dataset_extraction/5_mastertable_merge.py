#!/usr/bin/env python3
"""
create_mastertable.py

Merge SNP‐ and RNA‐feature tables (each including clinical + moca_change)
into one mastertable for GCN training.
"""

import os
import pandas as pd
import joblib
import numpy as np

def log2_plus_one(X):
    return np.log2(X + 1)

def main():
    # Paths to your per‐modality feature tables:
    snp_csv = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/snp_train_transformed_features.csv"
    rna_csv = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/rna_seq_train_transformed_features.csv"

    test_csv = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/data_splits/initial_split/test_data.csv"
    df_test = pd.read_csv(test_csv, dtype={'PATNO': str}).set_index('PATNO')
    # Load them (PATNO as index)
    df_snp = pd.read_csv(snp_csv, dtype={'PATNO': str}).set_index('PATNO')
    df_rna = pd.read_csv(rna_csv, dtype={'PATNO': str}).set_index('PATNO')

    # Sanity check: both must have the same clinical columns+label
    clinical_and_label = ['age_at_visit', 'SEX_M', 'EDUCYRS', 'moca_change']

    # Merge on PATNO (inner join ensures we keep only subjects with both sets)
    df_master = df_snp.drop(columns=clinical_and_label, errors='ignore').join(df_rna, how='inner')    # Note: we drop the duplicated clinical+label columns from the RNA side

    # --- Verification of PATNO merge correctness ---
    # Ensure index (PATNO) is unique
    assert df_master.index.is_unique, "Duplicate PATNOs found in merged mastertable"

    # Check that every PATNO present in both modality tables is in df_master
    common_ids = set(df_snp.index).intersection(df_rna.index)
    missing_ids = common_ids - set(df_master.index)
    assert not missing_ids, f"These PATNOs common to SNP and RNA are missing after merge: {missing_ids}"

    # Print final counts
    print(f"Verified merge: {len(df_master)} samples, matching {len(common_ids)} overlapping PATNOs")
    # --- End verification ---

    df_master['split'] = 'train'
    df_test['split'] = 'test'

    df_master = pd.concat([df_master, df_test], axis=0, join="inner")
    print("Mastertable with test rows now shape:", df_master.shape)

    # Step: Apply saved preprocessing only to test rows
    test_mask = df_master['split'] == 'test'

    # Load the fitted preprocessors
    rna_and_clin_prep = joblib.load("/Users/nickq/Documents/Pioneer Academics/Research_Project/data/preprocessing_pipeline/rna_and_clin_prep.joblib")
    snp_prep = joblib.load("/Users/nickq/Documents/Pioneer Academics/Research_Project/data/preprocessing_pipeline/snp_prep.joblib")

    # --- Auto-adjust and apply SNP+clinical preprocessor ---
    # Get the original column lists.
    expected_snp = list(snp_prep.feature_names_in_)
    # Compute intersection with current mastertable
    avail_snp = [c for c in expected_snp if c in df_master.columns]




    print(f"  [DEBUG] SNP transformer expected {len(expected_snp)} cols, {len(avail_snp)} available")
    print("  [DEBUG] Pre-transform test SNP slice shape:", df_master.loc[test_mask, avail_snp].shape)

    # Rebuild the transformer list to only include available columns per branch
    orig = snp_prep.transformers_
    new_transformers = []
    for name, transformer, cols in orig:
        if name == 'remainder':
            new_transformers.append((name, transformer, cols))
        else:
            kept = [c for c in cols if c in avail_snp]
            new_transformers.append((name, transformer, kept))
    # Apply the updated list
    snp_prep.set_params(transformers=new_transformers)
    # Also update the transformer's known feature_names to only the available ones
    snp_prep.feature_names_in_ = np.array(avail_snp)
    # Transform test rows
    df_master.loc[test_mask, avail_snp] = snp_prep.transform(df_master.loc[test_mask, avail_snp])
    print("  [DEBUG] Post-transform test SNP slice shape:", df_master.loc[test_mask, avail_snp].shape)

    # --- Auto-adjust and apply RNA preprocessor ---
    expected_rna = list(rna_and_clin_prep.feature_names_in_)
    avail_rna = []
    # Guarantee clinical continuous/binary columns are in avail_rna
    for col in ['age_at_visit', 'EDUCYRS', 'SEX_M']:
        if col in df_master.columns and col not in avail_snp:
            avail_rna.append(col)

    avail_rna    = [c for c in expected_rna if c in df_master.columns]



    print(f"  [DEBUG] RNA + clin transformer expected {len(expected_rna)} cols, {len(avail_rna)} available")
    print("  [DEBUG] Pre-transform test RNA slice shape:", df_master.loc[test_mask, avail_rna].shape)

    orig = rna_and_clin_prep.transformers_
    new_transformers = []
    for name, transformer, cols in orig:
        if name == 'remainder':
            new_transformers.append((name, transformer, cols))
        else:
            kept = [c for c in cols if c in avail_rna]
            new_transformers.append((name, transformer, kept))
    # Apply the updated list
    rna_and_clin_prep.set_params(transformers=new_transformers)
    # Also update the RNA transformer's feature_names_in_ to only available columns
    rna_and_clin_prep.feature_names_in_ = np.array(avail_rna)
    df_master.loc[test_mask, avail_rna] = rna_and_clin_prep.transform(df_master.loc[test_mask, avail_rna])
    print("  [DEBUG] Post-transform test RNA slice shape:", df_master.loc[test_mask, avail_rna].shape)

    print(f"Applied SNP+clinical ({len(avail_snp)} cols) and RNA ({len(avail_rna)} cols) preprocessing to test split.")

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