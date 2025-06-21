#!/usr/bin/env python3
"""
SNF Fusion Pipeline
-------------------
Loads preprocessed SNP and RNA feature tables, standardizes each block,
constructs sample-sample affinity graphs using squared-Euclidean distances,
performs Similarity Network Fusion (SNF), and saves the fused network.
"""


import os
import numpy as np
import pandas as pd
from snf import compute

# -----------------------------------------------------------------------------
# User-configurable output directory for fused networks
# -----------------------------------------------------------------------------
OUTPUT_FUSED_DIR = '/Users/nickq/Documents/Pioneer Academics/Research_Project/data/fused_datasets'

def load_data():
    # Adjust these paths to point to your final unprocessed CSVs
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'data', 'final_datasets_unprocessed')
    geno_path = os.path.join(base_dir, '/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/snp_train_transformed_features.csv')
    rna_path  = os.path.join(base_dir, '/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/lasso_output/rna_seq_train_transformed_features.csv')

    df_geno = pd.read_csv(geno_path, index_col='PATNO')
    df_rna  = pd.read_csv(rna_path,  index_col='PATNO')

    return df_geno, df_rna

def extract_features(df, exclude_cols=None):
    """Return a NumPy array of features by dropping specified columns."""
    if exclude_cols is None:
        exclude_cols = ['age_at_visit', 'SEX_M', 'EDUCYRS', 'moca_change']
    feat_df = df.drop(columns=exclude_cols, errors="ignore")
    return feat_df, feat_df.values


def build_affinity(X, K=20, mu=0.5):
    """Compute sample-sample affinity graph using SNFpy make_affinity."""
    # Directly use make_affinity on the data matrix
    W = compute.make_affinity(X, metric='sqeuclidean', K=K, mu=mu)
    return W

def fuse_networks(graphs, K=20, t=10):
    """Fuse a list of affinity matrices via SNF."""
    return compute.snf(graphs, K=K, t=t)

def save_fused(W, out_path=None):
    """Save fused network as NumPy array and CSV."""
    if out_path is None:
        # Use user-defined directory
        out_dir = OUTPUT_FUSED_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'W_fused.npy')
    np.save(out_path, W)
    # Also save as CSV for inspection
    csv_path = out_path.replace('.npy', '.csv')
    pd.DataFrame(W, index=None, columns=None).to_csv(csv_path, index=False)
    print(f"Fused network saved to {out_path} and {csv_path}")

def main():
    # Parameters
    K  = 20      # number of neighbors
    mu = 0.5     # kernel scaling factor
    t  = 10      # number of diffusion iterations

    # Load data
    df_geno, df_rna = load_data()

    # Extract features (already standardized upstream)
    _, X_geno = extract_features(df_geno)
    _, X_rna  = extract_features(df_rna)
    Xg = X_geno
    Xr = X_rna

    # Build affinity graphs
    print("Building SNP affinity...")
    W_geno = build_affinity(Xg, K=K, mu=mu)
    print("Building RNA affinity...")
    W_rna  = build_affinity(Xr, K=K, mu=mu)

    # Fuse networks
    print("Fusing networks with SNF...")
    W_fused = fuse_networks([W_geno, W_rna], K=K, t=t)

    # Save results
    save_fused(W_fused)

if __name__ == "__main__":
    main()