#!/usr/bin/env python3
"""
Merge SNP and RNA modality feature tables into a mastertable for GCN training.
"""

import os
import pandas as pd
import joblib
import numpy as np
import argparse

def log2_plus_one(X):
    return np.log2(X + 1)

def main():

    parser = argparse.ArgumentParser(description="RNA-seq Feature Selection Pipeline")
    parser.add_argument('--threshold', type=int, choices={-2, -3, -4, -5}, required=True, help="Threshold used in file names")
    args = parser.parse_args()


    for fold in range(5):
        # Input feature tables
        snp_csv = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/final_datasets_processed/lasso_output/snp_train_transformed_features_fold_{fold}_thresh_{args.threshold}.csv"
        rna_csv = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/final_datasets_processed/lasso_output/rna_seq_train_transformed_features_fold_{fold}_thresh_{args.threshold}.csv"

        test_csv = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/data_splits/cv_folds/test_fold_{fold}_thresh_{args.threshold}.csv"
        df_test = pd.read_csv(test_csv, dtype={'PATNO': str}).set_index('PATNO')
        # Load with PATNO as index
        df_snp = pd.read_csv(snp_csv, dtype={'PATNO': str}).set_index('PATNO')
        df_rna = pd.read_csv(rna_csv, dtype={'PATNO': str}).set_index('PATNO')

        # Shared clinical and label columns
        clinical_and_label = ['age_at_visit', 'SEX_M', 'EDUCYRS', 'label']

        # Inner join on PATNO (only overlapping samples)
        df_master = df_snp.drop(columns=clinical_and_label, errors='ignore').join(df_rna, how='inner')    # Note: we drop the duplicated clinical+label columns from the RNA side

        # Verify merge consistency
        # Confirm unique PATNOs
        assert df_master.index.is_unique, "Duplicate PATNOs found in merged mastertable"

        # Check no overlapping PATNOs are lost
        common_ids = set(df_snp.index).intersection(df_rna.index)
        missing_ids = common_ids - set(df_master.index)
        assert not missing_ids, f"These PATNOs common to SNP and RNA are missing after merge: {missing_ids}"

        # Confirm merge summary
        print(f"Verified merge: {len(df_master)} samples, matching {len(common_ids)} overlapping PATNOs")

        df_master['split'] = 'train'
        df_test['split'] = 'test'

        df_master = pd.concat([df_master, df_test], axis=0, join="inner")
        print("Mastertable with test rows now shape:", df_master.shape)

        # Apply test-time preprocessing
        test_mask = df_master['split'] == 'test'

        # Load fitted preprocessors
        rna_and_clin_prep = joblib.load(f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/preprocessing_pipeline/rna_and_clin_prep_fold_{fold}_thresh_{args.threshold}.joblib")
        snp_prep = joblib.load(f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/preprocessing_pipeline/snp_prep_fold_{fold}_thresh_{args.threshold}.joblib")

        # Adjust and apply SNP preprocessor
        # Expected SNP columns
        expected_snp = list(snp_prep.feature_names_in_)
        # Retain only available SNP columns
        avail_snp = [c for c in expected_snp if c in df_master.columns]




        print(f"  [DEBUG] SNP transformer expected {len(expected_snp)} cols, {len(avail_snp)} available")
        print("  [DEBUG] Pre-transform test SNP slice shape:", df_master.loc[test_mask, avail_snp].shape)

        # Rebuild transformer with available SNP columns
        orig = snp_prep.transformers_
        new_transformers = []
        for name, transformer, cols in orig:
            if name == 'remainder':
                new_transformers.append((name, transformer, cols))
            else:
                kept = [c for c in cols if c in avail_snp]
                new_transformers.append((name, transformer, kept))
        # Update transformer config
        snp_prep.set_params(transformers=new_transformers)
        # Update feature names
        snp_prep.feature_names_in_ = np.array(avail_snp)
        # Apply SNP transform to test
        df_master.loc[test_mask, avail_snp] = snp_prep.transform(df_master.loc[test_mask, avail_snp])
        print("  [DEBUG] Post-transform test SNP slice shape:", df_master.loc[test_mask, avail_snp].shape)

        # Adjust and apply RNA preprocessor
        expected_rna = list(rna_and_clin_prep.feature_names_in_)
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
        # Update RNA transformer config
        rna_and_clin_prep.set_params(transformers=new_transformers)
        # Update RNA feature names
        rna_and_clin_prep.feature_names_in_ = np.array(avail_rna)
        df_master.loc[test_mask, avail_rna] = rna_and_clin_prep.transform(df_master.loc[test_mask, avail_rna])
        print("  [DEBUG] Post-transform test RNA slice shape:", df_master.loc[test_mask, avail_rna].shape)

        print(f"Applied SNP+clinical ({len(avail_snp)} cols) and RNA ({len(avail_rna)} cols) preprocessing to test split.")

        # Summary info
        print("Mastertable shape:", df_master.shape)
        print("Columns breakdown:")
        print(" • Clinical/label:", clinical_and_label)
        print(" • SNP features:", [c for c in df_snp.columns if c not in clinical_and_label][:5], "...",
              f"({len(df_snp.columns) - len(clinical_and_label)} total)")
        print(" • RNA features:", [c for c in df_rna.columns if c not in clinical_and_label][:5], "...",
              f"({len(df_rna.columns) - len(clinical_and_label)} total)")

        # Save mastertable
        out_path = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/fused_datasets/final_mastertable_fold_{fold}_thresh_{args.threshold}.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_master.to_csv(out_path, index=True)
        print(f"Mastertable written to {out_path}")


if __name__ == "__main__":
    main()