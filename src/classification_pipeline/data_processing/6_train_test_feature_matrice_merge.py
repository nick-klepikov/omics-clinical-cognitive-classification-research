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
        # Load data
        snp_csv = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/final_datasets_processed/lasso_output/snp_train_transformed_features_fold_{fold}_thresh_{args.threshold}.csv"
        rna_csv = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/final_datasets_processed/lasso_output/rna_seq_train_transformed_features_fold_{fold}_thresh_{args.threshold}.csv"

        test_csv = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/data_splits/cv_folds/test_fold_{fold}_thresh_{args.threshold}.csv"
        df_test = pd.read_csv(test_csv, dtype={'PATNO': str}).set_index('PATNO')
        df_snp = pd.read_csv(snp_csv, dtype={'PATNO': str}).set_index('PATNO')
        df_rna = pd.read_csv(rna_csv, dtype={'PATNO': str}).set_index('PATNO')

        clinical_and_label = ['age_at_visit', 'SEX_M', 'EDUCYRS', 'label']

        # Merge SNP and RNA features
        df_master = df_snp.drop(columns=clinical_and_label, errors='ignore').join(df_rna, how='inner')

        assert df_master.index.is_unique, "Duplicate PATNOs found in merged mastertable"
        common_ids = set(df_snp.index).intersection(df_rna.index)
        missing_ids = common_ids - set(df_master.index)
        assert not missing_ids, f"These PATNOs common to SNP and RNA are missing after merge: {missing_ids}"

        df_master['split'] = 'train'
        df_test['split'] = 'test'

        df_master = pd.concat([df_master, df_test], axis=0, join="inner")

        # Preprocess test split
        test_mask = df_master['split'] == 'test'

        rna_and_clin_prep = joblib.load(f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/preprocessing_pipeline/rna_and_clin_prep_fold_{fold}_thresh_{args.threshold}.joblib")
        snp_prep = joblib.load(f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/preprocessing_pipeline/snp_prep_fold_{fold}_thresh_{args.threshold}.joblib")

        expected_snp = list(snp_prep.feature_names_in_)
        avail_snp = [c for c in expected_snp if c in df_master.columns]

        orig = snp_prep.transformers_
        new_transformers = []
        for name, transformer, cols in orig:
            if name == 'remainder':
                new_transformers.append((name, transformer, cols))
            else:
                kept = [c for c in cols if c in avail_snp]
                new_transformers.append((name, transformer, kept))
        snp_prep.set_params(transformers=new_transformers)
        snp_prep.feature_names_in_ = np.array(avail_snp)
        df_master.loc[test_mask, avail_snp] = snp_prep.transform(df_master.loc[test_mask, avail_snp])

        expected_rna = list(rna_and_clin_prep.feature_names_in_)
        avail_rna    = [c for c in expected_rna if c in df_master.columns]

        orig = rna_and_clin_prep.transformers_
        new_transformers = []
        for name, transformer, cols in orig:
            if name == 'remainder':
                new_transformers.append((name, transformer, cols))
            else:
                kept = [c for c in cols if c in avail_rna]
                new_transformers.append((name, transformer, kept))
        rna_and_clin_prep.set_params(transformers=new_transformers)
        rna_and_clin_prep.feature_names_in_ = np.array(avail_rna)
        df_master.loc[test_mask, avail_rna] = rna_and_clin_prep.transform(df_master.loc[test_mask, avail_rna])

        # Save mastertable
        out_path = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/fused_datasets/final_mastertable_fold_{fold}_thresh_{args.threshold}.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_master.to_csv(out_path, index=True)

if __name__ == "__main__":
    main()