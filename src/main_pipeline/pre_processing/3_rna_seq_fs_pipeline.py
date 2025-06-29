import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from smogn.smoter import smoter
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import joblib

# Log2(+1) transformation for gene features
def log2_plus_one(X):
    return np.log2(X + 1)

class PearsonFilter(BaseEstimator, TransformerMixin):
    """
    Select top k features by absolute Pearson correlation with the target.
    """
    def __init__(self, k=1000):
        self.k = k

    def fit(self, X, y):
        # Compute absolute Pearson r for each feature column
        # X: ndarray of shape (n_samples, n_features)
        # y: array of shape (n_samples,)
        corrs = []
        for i in range(X.shape[1]):
            # Compute correlation between feature i and target
            r = np.corrcoef(X[:, i], y)[0, 1]
            corrs.append(abs(r))
        self.scores_ = np.array(corrs)
        # Select indices of top k features
        self.selected_idx_ = np.argsort(self.scores_)[::-1][:self.k]
        return self

    def transform(self, X):
        # Subset X to only the selected feature columns
        return X[:, self.selected_idx_]

class EnLassoTransformer(BaseEstimator, TransformerMixin):
    """
    Perform SMOGN oversampling on 'moca_change' and bootstrapped Lasso to rank features.
    Uses 'moca_change' as the target column name.
    """
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
        bins  = [-np.inf, -3, -1, 0, np.inf]
        labels = [0, 1, 2, 3]
        y_bin = pd.cut(df['moca_change'], bins=bins, labels=labels).astype(int)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Accumulators for selection frequency and weight sums
        selection_counts = np.zeros(n_feats)
        weight_sums      = np.zeros(n_feats)

        # 5-fold stratified SMOGN + Lasso
        for train_idx, _ in skf.split(X, y_bin):
            # Build fold-specific DataFrame and reset index so positional iloc calls work correctly
            df_fold = df.iloc[train_idx].reset_index(drop=True)

            # Oversample extremes once per fold via SMOGN
            df_res = smoter(data=df_fold, y='moca_change')
            print("SMOGN oversampling complete for current fold.")

            # Bootstrap from the SMOGN output
            for i in range(self.n_runs):
                print(f"  Bootstrap iteration {i+1}/{self.n_runs} for current fold")
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
                weight_sums      += coef

        # Compute Importance Score per Yang et al.
        FSq = selection_counts / (5 * self.n_runs)
        FSw = weight_sums      / (5 * self.n_runs)
        IS  = 0.5 * (FSq + FSw)

        # Select top_k features by descending IS
        self.selected_idx_ = np.argsort(IS)[::-1][:self.top_k]
        return self

    def transform(self, X):
        """
        Subset the input feature matrix to only the previously selected features.
        """
        return X[:, self.selected_idx_]

# 1) Load your training DataFrame
df_train = pd.read_csv('/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/rna_plus_clinical_final.csv')
ids = df_train['PATNO'].values

# 2) Identify columns
target_col = 'moca_change'
binary_cols = ['SEX_M']  # sex flag
continuous_cols = ['age_at_visit', 'EDUCYRS']
# (add any other clinical/cognitive continuous variables here)
gene_cols = [c for c in df_train.columns if c.startswith('ENSG')]

# Full ordered list of feature names before filtering
all_features = continuous_cols + binary_cols + gene_cols

# 3) Build ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    # a) Impute and scale continuous features
    ('cont', Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ]), continuous_cols),

    # b) Impute binary flags (no scaling)
    ('bin', SimpleImputer(strategy='most_frequent'), binary_cols),

    # c) Impute gene features, log2(+1) transform, then z-score
    ('genes', Pipeline([
        ('imp0',   SimpleImputer(strategy='constant', fill_value=0)),
        ('log2',   FunctionTransformer(log2_plus_one, validate=True)),
        ('scale',  StandardScaler())
    ]), gene_cols),
],
    remainder='drop')  # drop any other columns

# 4) Wrap into a Pipeline
fs_pipe = Pipeline([
    ('pre',      preprocessor),  # impute, log2, scale
    ('pearson',  PearsonFilter(k=1000)),  # select top 1000 by Pearson correlation
    ('enlasso',  EnLassoTransformer(n_runs=100, alpha=0.01, top_k=20, random_state=42)),  # SMOGN + Lasso stability selection with fixed seed
])

# 5) Fit
X_train = df_train[continuous_cols + binary_cols + gene_cols]
y_train = df_train[target_col]
fs_pipe.fit(X_train, y_train)

# Extract and display the final selected feature names and their ranks
pearson_idx = fs_pipe.named_steps['pearson'].selected_idx_
# Features after Pearson filtering
feat_after_pearson = [all_features[i] for i in pearson_idx]

enlasso_idx = fs_pipe.named_steps['enlasso'].selected_idx_
selected_features = [feat_after_pearson[i] for i in enlasso_idx]

# Clinical columns
rna_only_cont = [c for c in ['age_at_visit', 'EDUCYRS'] if c in selected_features]
rna_only_binary = [c for c in ['SEX_M'] if c in selected_features]
# rna-seq columns selected by EnLasso
rna_only_rna_seq = [c for c in selected_features if c not in rna_only_cont + rna_only_binary]

# Build transformer on exactly those columns
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

# Fit on the training subset of those columns
X_train_sel = df_train[rna_only_cont + rna_only_binary + rna_only_rna_seq]
rna_and_clin_pre.fit(X_train_sel)

# Export this fitted SNP-only preprocessor
rna_path = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/preprocessing_pipeline/rna_and_clin_prep.joblib"
joblib.dump(rna_and_clin_pre, rna_path)
print(f"Saved RNA-only preprocessor to {rna_path}")



# Transform training data
X_train_trans = fs_pipe.transform(X_train)
# Build a DataFrame for transformed features with column names
df_train_trans = pd.DataFrame(X_train_trans, columns=selected_features)
# Prepend the PATNO column
df_train_trans.insert(0, 'PATNO', ids)
df_train_trans.insert(1, 'moca_change', y_train)

# Save transformed training data to CSV
df_train_trans.to_csv('/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/rna_seq_train_transformed_features.csv', index=False)

# Build DataFrame of final selected features
df_selected = pd.DataFrame({
    'feature_name': selected_features,
    'rank': np.arange(1, len(selected_features) + 1)
})

print("Top selected features:")
print(df_selected.to_string(index=False))
# Save the selected features list to CSV for downstream use
df_selected.to_csv('/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/rna_seq_selected_features.csv', index=False)
print("Selected features saved to rna_seq_selected_features.csv")