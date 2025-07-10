import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SVMSMOTE
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import joblib
import argparse


class EnLassoTransformer(BaseEstimator, TransformerMixin):
    """
    Use SVMSMOTE and bootstrapped Lasso to rank and select features.
    """
    def __init__(self, n_runs=100, alpha=0.01, top_k=200, random_state=None):
        self.n_runs = n_runs
        self.alpha = alpha
        self.top_k = top_k
        self.random_state = random_state

    def fit(self, X, y):
        n_feats = X.shape[1]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        selection_counts = np.zeros(n_feats)
        weight_sums      = np.zeros(n_feats)

        for train_idx, _ in skf.split(X, y):
            X_fold = X[train_idx]
            y_fold = y[train_idx]

            svmsmote = SVMSMOTE(random_state=self.random_state)
            X_res, y_res = svmsmote.fit_resample(X_fold, y_fold)
            print("SVMSMOTE oversampling complete for current fold.")

            for i in range(self.n_runs):
                print(f"  Bootstrap iteration {i+1}/{self.n_runs} for current fold")
                boot_idx = resample(np.arange(len(y_res)), replace=True, n_samples=len(y_res), random_state=42)
                X_boot = X_res[boot_idx]
                y_boot = y_res[boot_idx]

                lasso = Lasso(alpha=self.alpha, max_iter=50000, tol=1e-4)
                lasso.fit(X_boot, y_boot)
                coef = np.abs(lasso.coef_)

                selection_counts += (coef > 0).astype(int)
                weight_sums      += coef

        FSq = selection_counts / (5 * self.n_runs)
        FSw = weight_sums      / (5 * self.n_runs)
        IS  = 0.5 * (FSq + FSw)

        self.selected_idx_ = np.argsort(IS)[::-1][:self.top_k]
        return self

    def transform(self, X):
        return X[:, self.selected_idx_]


def main():
    parser = argparse.ArgumentParser(description="RNA-seq Feature Selection Pipeline")
    parser.add_argument('--threshold', type=int, choices={-2, -3, -4}, required=True, help="Threshold used in file names")
    args = parser.parse_args()
    threshold = args.threshold

    for fold in range(5):
        # 1) Load SNP + clinical training data direct from combined CSV
        input_path = f'/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_unprocessed/geno_plus_clinical_fold_{fold}_thresh_{threshold}.csv'
        df_train = pd.read_csv(
            input_path,
            dtype={'PATNO': str}
        )
        print(f"[DEBUG] Fold {fold} Threshold {threshold}: Loaded {len(df_train)} training rows from {input_path}")
        ids = df_train['PATNO'].values

        # 2) Identify columns
        target_col = 'label'
        binary_cols = ['SEX_M']  # sex flag
        continuous_cols = ['age_at_visit', 'EDUCYRS']

        # Define RNA-seq gene expression columns (start with 'ENSG')
        rna_cols = [c for c in df_train.columns if c.startswith('ENSG')]
        # Exclude RNA-seq columns so only SNP dosage features remain
        df_train = df_train.drop(columns=rna_cols)
        # Report number of features after dropping RNA-seq columns (excluding PATNO)
        num_feats = df_train.shape[1] - 1  # subtract PATNO
        print(f"[INFO] Number of remaining feature columns (excluding PATNO): {num_feats}")

        # Define SNP dosage columns as any remaining feature columns
        excluded = set(continuous_cols + binary_cols + [target_col, 'PATNO'])
        snp_cols = [c for c in df_train.columns if c not in excluded]

        # Full ordered list of feature names after dropping RNA-seq columns
        all_features = continuous_cols + binary_cols + snp_cols

        # 3) Build ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            # a) Impute and scale continuous features
            ('cont', Pipeline([
                ('impute', SimpleImputer(strategy='mean')),
                ('scale', StandardScaler())
            ]), continuous_cols),

            # b) Impute binary flags (no scaling)
            ('bin', SimpleImputer(strategy='most_frequent'), binary_cols),

            # c) Impute SNP dosage features, then z-score
            ('snps', Pipeline([
                ('imp0', SimpleImputer(strategy='constant', fill_value=0)),
                ('scale', StandardScaler())
            ]), snp_cols),
        ],
            remainder='drop')  # drop any other columns

        # 4) Wrap into a Pipeline
        fs_pipe = Pipeline([
            ('pre', preprocessor),
            ('enlasso', EnLassoTransformer(n_runs=100, alpha=0.01, top_k=20, random_state=42)),
        ])

        # 5) Fit
        X_train = df_train[continuous_cols + binary_cols + snp_cols]
        y_train = df_train[target_col]
        fs_pipe.fit(X_train, y_train)

        # Extract and display the final selected feature names and their ranks
        enlasso_idx = fs_pipe.named_steps['enlasso'].selected_idx_
        selected_features = [all_features[i] for i in enlasso_idx]

        # Clinical columns
        snp_only_cont = [c for c in ['age_at_visit', 'EDUCYRS'] if c in selected_features]
        snp_only_binary = [c for c in ['SEX_M'] if c in selected_features]
        # SNP columns selected by EnLasso
        snp_only_snps = [c for c in selected_features if c not in snp_only_cont + snp_only_binary]

        # Build transformer on exactly those columns
        snp_only_pre = ColumnTransformer(transformers=[
            ('snps', Pipeline([
                ('imp0', SimpleImputer(strategy='constant', fill_value=0)),
                ('scale', StandardScaler())
            ]), snp_only_snps),
        ], remainder='drop')

        # Fit on the training subset of those columns
        X_train_sel = df_train[snp_only_snps]
        snp_only_pre.fit(X_train_sel)

        # Export this fitted SNP-only preprocessor
        snp_only_path = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/preprocessing_pipeline/snp_prep_fold_{fold}_thresh_{threshold}.joblib"
        joblib.dump(snp_only_pre, snp_only_path)
        print(f"Fold {fold} Threshold {threshold}: Saved SNP-only preprocessor to {snp_only_path}")

        # Transform training data
        X_train_trans = fs_pipe.transform(X_train)
        # Build a DataFrame for transformed features with column names
        df_train_trans = pd.DataFrame(X_train_trans, columns=selected_features)
        # Prepend the PATNO column
        df_train_trans.insert(0, 'PATNO', ids)
        df_train_trans.insert(1, 'label', y_train)

        # Save transformed training data to CSV
        output_transformed_path = f'/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/snp_train_transformed_features_fold_{fold}_thresh_{threshold}.csv'
        df_train_trans.to_csv(output_transformed_path, index=False)
        print(f"Fold {fold} Threshold {threshold}: Saved transformed training data to {output_transformed_path}")


if __name__ == "__main__":
    main()