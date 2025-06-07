#!/usr/bin/env python3
"""
Pure-NumPy genotype + clinical preprocessing script.
"""

import csv
import numpy as np

# === CONFIGURATION ===
INPUT_CSV  = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_unprocessed/geno_plus_clinical.csv"
OUTPUT_CSV = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/geno_plus_clinical_final_numpy.csv"

def main():
    # 1) Read header to identify columns
    with open(INPUT_CSV, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
    # Identify columns
    patno_idx = header.index("PATNO")
    sex_idx   = header.index("SEX")
    cont_idxs = [header.index(c) for c in ("age_at_visit","EDUCYRS","moca","moca_12m") if c in header]
    # SNP cols = all others except PATNO, SEX, and cont
    exclude = {patno_idx, sex_idx} | set(cont_idxs)
    snp_idxs = [i for i in range(len(header)) if i not in exclude]

    # Columns to load for numeric data: clinical continuous + SNPs
    load_cols = cont_idxs + snp_idxs

    # 2) Load PATNO and SEX columns
    patnos = []
    sex = []
    with open(INPUT_CSV, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            patnos.append(row[patno_idx])
            sex.append(int(row[sex_idx] == "M"))
    sex = np.array(sex, dtype=np.int8)

    # 3) Load continuous clinical and SNP data, allowing blank fields as NaN
    data = np.genfromtxt(
        INPUT_CSV,
        delimiter=",",
        skip_header=1,
        dtype=np.float32,
        usecols=load_cols,
        missing_values="",
        filling_values=np.nan,
        autostrip=True
    )
    # data shape: (n_samples, n_cont + n_snps)

    # 4) Impute mean for NaNs in each column
    col_means = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_means, inds[1])

    # 5) Z-score normalization for continuous columns (all in data)
    col_stds = data.std(axis=0, ddof=0)
    # avoid division by zero
    col_stds[col_stds == 0] = 1.0
    data = (data - col_means) / col_stds

    # 6) Concatenate sex column at the end
    # Expand sex to shape (n_samples,1)
    sex_col = sex.reshape(-1,1).astype(np.float32)
    full = np.hstack([data, sex_col])

    # 7) Build output header
    feature_names = [header[i] for i in cont_idxs] + [header[i] for i in snp_idxs] + ["SEX_M"]
    out_header = ["PATNO"] + feature_names

    # 8) Write to CSV
    with open(OUTPUT_CSV, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        for pid, row in zip(patnos, full):
            writer.writerow([pid] + row.tolist())

    print(f"Wrote NumPy-processed genotype+clinical data → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()