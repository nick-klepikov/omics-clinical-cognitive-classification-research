import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SVMSMOTE
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import joblib

# Log2(+1) transform for gene data
def log2_plus_one(X):
    return np.log2(X + 1)

class FisherFilter(BaseEstimator, TransformerMixin):
    """
    Select top k features by Fisher Discriminant Ratio for binary classification.
    """
    def __init__(self, k=1000, epsilon=1e-8):
        self.k = k
        self.epsilon = epsilon

    def fit(self, X, y):
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("FisherFilter requires binary target.")
        class0, class1 = classes

        fdr_scores = []
        for i in range(X.shape[1]):
            x = X[:, i]
            x0 = x[y == class0]
            x1 = x[y == class1]
            mu0 = np.nanmean(x0)
            mu1 = np.nanmean(x1)
            var0 = np.nanvar(x0)
            var1 = np.nanvar(x1)
            denom = var0 + var1 + self.epsilon
            fdr = ((mu1 - mu0) ** 2) / denom
            fdr_scores.append(fdr)
        self.scores_ = np.array(fdr_scores)
        self.selected_idx_ = np.argsort(self.scores_)[::-1][:self.k]
        return self

    def transform(self, X):
        return X[:, self.selected_idx_]

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

    # Repeat for each fold
    for fold in range(5):
        # Load training data (with threshold and fold in filename)
        train_path = f'/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/rna_plus_clinical_final_fold_{fold}_thresh_{threshold}.csv'
        df_train = pd.read_csv(train_path)
        ids = df_train['PATNO'].values

        # Define columns
        target_col = 'label'
        binary_cols = ['SEX_M']
        continuous_cols = ['age_at_visit', 'EDUCYRS']
        gene_cols = [c for c in df_train.columns if c.startswith('ENSG')]

        all_features = continuous_cols + binary_cols + gene_cols

        # Preprocessing pipeline for features
        preprocessor = ColumnTransformer(transformers=[
            ('cont', Pipeline([
                ('impute', SimpleImputer(strategy='mean')),
                ('scale', StandardScaler())
            ]), continuous_cols),

            ('bin', SimpleImputer(strategy='most_frequent'), binary_cols),

            ('genes', Pipeline([
                ('imp0',   SimpleImputer(strategy='constant', fill_value=0)),
                ('log2',   FunctionTransformer(log2_plus_one, validate=True)),
                ('scale',  StandardScaler())
            ]), gene_cols),
        ],
            remainder='drop')

        # Feature selection pipeline
        fs_pipe = Pipeline([
            ('pre',      preprocessor),
            ('fisher',  FisherFilter(k=1000)),
            ('enlasso',  EnLassoTransformer(n_runs=100, alpha=0.01, top_k=20, random_state=42)),
        ])

        # Fit pipeline
        X_train = df_train[continuous_cols + binary_cols + gene_cols]
        y_train = df_train[target_col]
        fs_pipe.fit(X_train, y_train)

        # Retrieve selected features after Fisher filter
        fisher_idx = fs_pipe.named_steps['fisher'].selected_idx_
        feat_after_fisher = [all_features[i] for i in fisher_idx]

        # Retrieve final selected features after EnLasso
        enlasso_idx = fs_pipe.named_steps['enlasso'].selected_idx_
        selected_features = [feat_after_fisher[i] for i in enlasso_idx]

        # Separate clinical and gene features selected
        rna_only_cont = [c for c in ['age_at_visit', 'EDUCYRS'] if c in selected_features]
        rna_only_binary = [c for c in ['SEX_M'] if c in selected_features]
        rna_only_rna_seq = [c for c in selected_features if c not in rna_only_cont + rna_only_binary]

        # Build preprocessor for selected features only
        rna_and_clin_pre = ColumnTransformer(transformers=[
            ('cont', Pipeline([
                ('impute', SimpleImputer(strategy='mean')),
                ('scale', StandardScaler())
            ]), rna_only_cont),
            ('bin', SimpleImputer(strategy='most_frequent'), rna_only_binary),
            ('genes', Pipeline([
                ('imp0', SimpleImputer(strategy='constant', fill_value=0)),
                ('log2', FunctionTransformer(log2_plus_one, validate=True)),
                ('scale', StandardScaler())
            ]), rna_only_rna_seq),
        ], remainder='drop')

        X_train_sel = df_train[rna_only_cont + rna_only_binary + rna_only_rna_seq]
        rna_and_clin_pre.fit(X_train_sel)

        # Save the fitted preprocessor
        rna_path = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/preprocessing_pipeline/rna_and_clin_prep_fold_{fold}_thresh_{threshold}.joblib"
        joblib.dump(rna_and_clin_pre, rna_path)
        print(f"[Fold {fold}] Saved RNA-only preprocessor to {rna_path}")

        # Transform training data with full pipeline
        X_train_trans = fs_pipe.transform(X_train)
        df_train_trans = pd.DataFrame(X_train_trans, columns=selected_features)
        df_train_trans.insert(0, 'PATNO', ids)
        df_train_trans.insert(1, 'label', y_train)

        # Save transformed features
        out_trans = f'/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/rna_seq_train_transformed_features_fold_{fold}_thresh_{threshold}.csv'
        df_train_trans.to_csv(out_trans, index=False)



if __name__ == "__main__":
    main()