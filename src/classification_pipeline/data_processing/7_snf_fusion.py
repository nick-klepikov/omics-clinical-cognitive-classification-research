"""
SNF Fusion Pipeline
-------------------
Loads preprocessed SNP and RNA feature tables, standardizes each block,
constructs sample-sample affinity graphs using squared-Euclidean distances,
performs Similarity Network Fusion (SNF), and saves the fused network.
"""


import os
import argparse
import numpy as np
import pandas as pd
from snf import compute

# Output directory

OUTPUT_FUSED_DIR = '/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/fused_datasets/affinity_matrices'

# Load modalities
def load_data(mastertable_path):
    df_master = pd.read_csv(mastertable_path, index_col='PATNO')
    clinical_and_label = ['age_at_visit','SEX_M','EDUCYRS','label','split']
    rna_cols = [c for c in df_master.columns if c.startswith('ENSG')]
    snp_cols = [c for c in df_master.columns if c not in clinical_and_label + rna_cols]
    df_snp = df_master[snp_cols]
    df_rna = df_master[rna_cols]
    return df_snp, df_rna

def extract_features(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['age_at_visit', 'SEX_M', 'EDUCYRS', 'label', "split"]
    feat_df = df.drop(columns=exclude_cols, errors="ignore")
    return feat_df, feat_df.values

# Compute affinity graph
def build_affinity(X, K=20, mu=0.5):
    W = compute.make_affinity(X, metric='sqeuclidean', K=K, mu=mu)
    return W

# Fuse affinity matrices
def fuse_networks(graphs, K=20, t=10):
    return compute.snf(graphs, K=K, t=t)

# Save network to disk
def save_fused(W, out_path=None):
    if out_path is None:
        out_dir = OUTPUT_FUSED_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'W_fused.npy')
    np.save(out_path, W)
    csv_path = out_path.replace('.npy', '.csv')
    pd.DataFrame(W, index=None, columns=None).to_csv(csv_path, index=False)

def save_single(W, type, out_path=None):
    if out_path is None:
        out_dir = OUTPUT_FUSED_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"W_{type}.npy")
    np.save(out_path, W)
    csv_path = out_path.replace('.npy', '.csv')
    pd.DataFrame(W, index=None, columns=None).to_csv(csv_path, index=False)

def main():

    parser = argparse.ArgumentParser(description="RNA-seq Feature Selection Pipeline")
    parser.add_argument('--threshold', type=int, choices={-2, -3, -4, -5}, required=True,
                        help="Threshold used in file names")
    args = parser.parse_args()

    # Run SNF for each fold

    K  = 20
    mu = 0.5
    t  = 10

    for fold in range(5):
        mastertable_path = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/fused_datasets/final_mastertable_fold_{fold}_thresh_{args.threshold}.csv"
        df_geno, df_rna = load_data(mastertable_path)

        _, X_geno = extract_features(df_geno)
        _, X_rna  = extract_features(df_rna)
        Xg = X_geno
        Xr = X_rna

        W_geno = build_affinity(Xg, K=K, mu=mu)
        W_rna  = build_affinity(Xr, K=K, mu=mu)

        W_fused = fuse_networks([W_geno, W_rna], K=K, t=t)

        out_dir = OUTPUT_FUSED_DIR
        os.makedirs(out_dir, exist_ok=True)
        fused_out_path = os.path.join(out_dir, f"W_fused_fold_{fold}_thresh_{args.threshold}.npy")
        geno_out_path = os.path.join(out_dir, f"W_geno_fold_{fold}_thresh_{args.threshold}.npy")
        rna_out_path = os.path.join(out_dir, f"W_rna_fold_{fold}_thresh_{args.threshold}.npy")

        save_fused(W_fused, out_path=fused_out_path)
        save_single(W_geno, "geno", out_path=geno_out_path)
        save_single(W_rna, "rna", out_path=rna_out_path)

if __name__ == "__main__":
    main()