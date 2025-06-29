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

def load_data(mastertable_path):
    # Load the merged mastertable containing train+val+test
    df_master = pd.read_csv(mastertable_path, index_col='PATNO')
    # Define clinical/label/Nepotism columns to exclude from modalities
    clinical_and_label = ['age_at_visit','SEX_M','EDUCYRS','moca_change','split']
    # Identify RNA columns (e.g., those starting with 'ENSG') and SNP columns
    rna_cols = [c for c in df_master.columns if c.startswith('ENSG')]
    snp_cols = [c for c in df_master.columns if c not in clinical_and_label + rna_cols]
    # Split into modality DataFrames
    df_snp = df_master[snp_cols]
    df_rna = df_master[rna_cols]
    return df_snp, df_rna

def extract_features(df, exclude_cols=None):
    """Return a NumPy array of features by dropping specified columns."""
    if exclude_cols is None:
        exclude_cols = ['age_at_visit', 'SEX_M', 'EDUCYRS', 'moca_change', "split"]
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

def save_single(W, type, out_path=None):
    """Save single modality network as NumPy array and CSV."""
    if out_path is None:
        # Use user-defined directory
        out_dir = OUTPUT_FUSED_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"W_{type}.npy")
    np.save(out_path, W)
    # Also save as CSV for inspection
    csv_path = out_path.replace('.npy', '.csv')
    pd.DataFrame(W, index=None, columns=None).to_csv(csv_path, index=False)
    print(f"Single modality {type} network saved to {out_path} and {csv_path}")

def main():
    # Parameters
    K  = 20      # number of neighbors
    mu = 0.5     # kernel scaling factor
    t  = 10      # number of diffusion iterations

    # Load modalities from the merged mastertable
    mastertable_path = '/Users/nickq/Documents/Pioneer Academics/Research_Project/data/fused_datasets/final_mastertable.csv'
    df_geno, df_rna = load_data(mastertable_path)

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
    save_single(W_geno, "geno")
    save_single(W_rna, "rna")

if __name__ == "__main__":
    main()