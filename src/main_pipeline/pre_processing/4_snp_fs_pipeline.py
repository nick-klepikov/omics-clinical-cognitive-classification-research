"""
SNP Feature Selection Pipeline:
- SMOGN + Bootstrapped Lasso on SNP dosage features.
- Target: moca_change.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from smogn.smoter import smoter
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import joblib


class EnLassoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_runs=100, alpha=0.01, top_k=200, random_state=None):
        """
        n_runs: number of SMOGN + Lasso repetitions to stabilize feature rankings.
        alpha: L1 regularization strength for Lasso.
        top_k: number of top features to select based on average coefficient magnitudes.
        random_state: seed for reproducibility of the entire procedure.
        """
        self.n_runs = n_runs
        self.alpha = alpha
        self.top_k = top_k
        self.random_state = random_state

    def fit(self, X, y):
        """
        X: ndarray of shape (n_samples, n_features)
        y: array-like of shape (n_samples,) with ΔMoCA continuous values.

        The random number generator 'rng' drives all randomness here,
        generating independent seeds for each iteration without affecting global random state.
        """
        # Number of features
        n_feats = X.shape[1]

        # SMOGN requires a DataFrame with named features and target column
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_feats)])
        df['moca_change'] = y  # assign provided target values directly

        # Bin ΔMoCA into four intervals based on value ranges:
        # 0 = rapid decline: Δ < -3
        # 1 = moderate decline: -3 ≤ Δ < -1
        # 2 = slight decline/no change: -1 ≤ Δ ≤ 0
        # 3 = improvement: Δ > 0
        bins = [-np.inf, -3, -1, 0, np.inf]
        labels = [0, 1, 2, 3]
        y_bin = pd.cut(df['moca_change'], bins=bins, labels=labels).astype(int)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Accumulators for selection frequency and weight sums
        selection_counts = np.zeros(n_feats)
        weight_sums = np.zeros(n_feats)

        # 5-fold stratified SMOGN + Lasso
        for train_idx, _ in skf.split(X, y_bin):
            # Build fold-specific DataFrame and reset index so positional iloc calls work correctly
            df_fold = df.iloc[train_idx].reset_index(drop=True)

            # Oversample extremes once per fold via SMOGN
            df_res = smoter(data=df_fold, y='moca_change')
            print("SMOGN oversampling complete for current fold.")

            # Bootstrap from the SMOGN output
            for i in range(self.n_runs):
                print(f"  Bootstrap iteration {i + 1}/{self.n_runs} for current fold")
                # Bootstrap sample from the SMOGN output
                boot_idx = resample(np.arange(len(df_res)), replace=True, n_samples=len(df_res), random_state=42)
                df_boot = df_res.iloc[boot_idx].reset_index(drop=True)

                # Separate features and target for bootstrap sample
                X_res = df_boot.drop(columns=['moca_change']).values
                y_res = df_boot['moca_change'].values

                # Increase iterations and loosen convergence tolerance to ensure convergence
                lasso = Lasso(alpha=self.alpha, max_iter=50000, tol=1e-4)
                lasso.fit(X_res, y_res)
                coef = np.abs(lasso.coef_)

                # Accumulate frequency of selection and weight sums
                selection_counts += (coef > 0).astype(int)
                weight_sums += coef

        # Compute Importance Score per Yang et al.
        FSq = selection_counts / (5 * self.n_runs)
        FSw = weight_sums / (5 * self.n_runs)
        IS = 0.5 * (FSq + FSw)

        # Select top_k features by descending IS
        self.selected_idx_ = np.argsort(IS)[::-1][:self.top_k]
        return self

    def transform(self, X):
        """
        Subset the input feature matrix to only the previously selected features.
        """
        return X[:, self.selected_idx_]


# 1) Load SNP + clinical training data direct from combined CSV
df_train = pd.read_csv(
    '/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_unprocessed/geno_plus_clinical.csv',
    dtype={'PATNO': str}
)
print(f"[DEBUG] Loaded {len(df_train)} training rows from geno_plus_clinical.csv")
ids = df_train['PATNO'].values

# 2) Identify columns
target_col = 'moca_change'
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
snp_only_path = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/preprocessing_pipeline/snp_prep.joblib"
joblib.dump(snp_only_pre, snp_only_path)
print(f"Saved SNP-only preprocessor to {snp_only_path}")

# Transform training data
X_train_trans = fs_pipe.transform(X_train)
# Build a DataFrame for transformed features with column names
df_train_trans = pd.DataFrame(X_train_trans, columns=selected_features)
# Prepend the PATNO column
df_train_trans.insert(0, 'PATNO', ids)
df_train_trans.insert(1, 'moca_change', y_train)

# Save transformed training data to CSV
df_train_trans.to_csv(
    '/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/snp_train_transformed_features.csv',
    index=False)

# Build DataFrame of final selected features
df_selected = pd.DataFrame({
    'feature_name': selected_features,
    'rank': np.arange(1, len(selected_features) + 1)
})

print("Top selected features:")
print(df_selected.to_string(index=False))
# Save the selected features list to CSV for downstream use
df_selected.to_csv(
    '/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/snp_selected_features.csv',
    index=False)
print("Selected features saved to rna_seq_selected_features.csv")
