import pandas as pd  # data_processing manipulation
import re  # regex for filename parsing
from pandas_plink import read_plink
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse

# Parse threshold argument for binarizing MoCA change
parser = argparse.ArgumentParser(description="Generate mastertable with given threshold")
parser.add_argument('--threshold', type=int, choices=[-2, -3, -4, -5],  required=True, help='Threshold for binarizing MoCA change')
args = parser.parse_args()

# ──────────────── 1) Clinical & Cognitive Data ────────────────
clinical_path = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/raw/clinical_and_cognitive_curated/PPMI_Curated_Data_Cut_Public_20250321.xlsx"
df_clin = pd.read_excel(
    clinical_path,
    sheet_name="20250310",
    dtype={"PATNO": str}
)

# Filter for baseline and 12-month, only "Sporadic PD"
if "EVENT_ID" in df_clin.columns:
    df_clin_bl = df_clin[(df_clin["EVENT_ID"] == "BL") & (df_clin["subgroup"] == "Sporadic PD")].copy()
    df_v04 = df_clin[(df_clin["EVENT_ID"] == "V04") & (df_clin["subgroup"] == "Sporadic PD")].copy()
    if "moca" in df_v04.columns:
        df_v04 = df_v04.rename(columns={"moca": "moca_12m"})
    v04_keep = ["PATNO"]
    if "moca_12m" in df_v04.columns:
        v04_keep.append("moca_12m")
    df_v04 = df_v04[v04_keep]
    # Merge baseline and 12m MoCA
    df_clin = pd.merge(
        df_clin_bl,
        df_v04,
        on="PATNO",
        how="left"
    )
    # Remove subjects missing MoCA at either timepoint
    df_clin = df_clin.dropna(subset=["moca", "moca_12m"]).reset_index(drop=True)
    df_clin["moca"] = pd.to_numeric(df_clin["moca"], errors="coerce")
    df_clin["moca_12m"] = pd.to_numeric(df_clin["moca_12m"], errors="coerce")
    df_clin["moca_change"] = df_clin["moca_12m"] - df_clin["moca"]
    df_clin["label"] = (df_clin["moca_change"] <= args.threshold).astype(int)

# Keep only relevant clinical columns
keep_cols = ["PATNO", "age_at_visit", "SEX", "EDUCYRS", "label"]
df_clin = df_clin[[c for c in keep_cols if c in df_clin.columns]].copy()

# Encode SEX as binary (1 = male)
if 'SEX' in df_clin.columns:
    df_clin["SEX_M"] = df_clin["SEX"].astype(int)
    df_clin = df_clin.drop(columns=["SEX"], errors="ignore")
    print("Encoded SEX → SEX_M column added")

print(f"[1] Clinical → {len(df_clin)} rows, {len(df_clin.columns)} columns after filtering")
clinic_patnos = set(df_clin["PATNO"])


# ──────────────── 2) Genotyping Data ────────────────
# Load IID→PATNO mapping
df_link = pd.read_csv(
    "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/raw/genetics/PPMI_244_Plink/ppmi_244_linklist.csv",
    dtype={"GP2sampleID": str, "PATNO": str, "COHORT": str})
clinic_iids = set(
    df_link.loc[df_link["PATNO"].isin(clinic_patnos), "GP2sampleID"]
)

# Load PLINK data_processing
(bim, fam, bed) = read_plink("/Users/nickq/Documents/Pioneer Academics/Research_Project/data/raw/genetics/PPMI_244_Plink/nonGR_LONI_PPMI_MAY2023")
# Filter samples to those with clinical data_processing
keep_samples = fam["iid"].isin(clinic_iids).values
fam = fam[keep_samples].reset_index(drop=True)
bed = bed[:, keep_samples]
full_mat = bed.compute()  # load genotype array into memory
sample_ids = fam.iid.values.tolist()
snp_ids = bim.snp.values.tolist()

# SNP QC: missingness, MAF, LD pruning
snp_index_map = {s: idx for idx, s in enumerate(bim.snp.values.tolist())}
batch_size = 10000
good_snps = []
for start in range(0, len(snp_ids), batch_size):
    print(f"[QC: missing+MAF] Batch {start}-{min(start+batch_size, len(snp_ids))} of {len(snp_ids)} SNPs")
    batch = snp_ids[start:start+batch_size]
    arr = full_mat[[snp_index_map[s] for s in batch], :].T
    df_batch = pd.DataFrame(arr, columns=batch)
    miss = df_batch.isna().mean()
    keep1 = miss[miss <= 0.05].index.tolist()
    p = df_batch[keep1].mean(skipna=True) / 2
    maf = np.minimum(p, 1-p)
    keep2 = maf[maf >= 0.01].index.tolist()
    good_snps.extend(keep2)
    del df_batch
good_snps = list(dict.fromkeys(good_snps))

# LD pruning
pruned = []
WINDOW, STEP, R2 = 50, 5, 0.2
for i in range(0, len(good_snps), STEP):
    print(f"[QC: LD prune] Window start {i} of {len(good_snps)} SNPs")
    window = good_snps[i:i+WINDOW]
    arr = full_mat[[snp_index_map[s] for s in window], :].T
    df_window = pd.DataFrame(arr, columns=window)
    col_means = df_window.mean()
    df_window = df_window.fillna(col_means)
    if len(window) < 2:
        pruned.extend(window)
        continue
    stds = df_window.std(ddof=0).replace(0, 1)
    G = (df_window - df_window.mean()) / stds
    corr2 = np.corrcoef(G.values, rowvar=False)**2
    removed = set()
    for j, s in enumerate(window):
        if s in removed:
            continue
        pruned.append(s)
        high = np.where(corr2[j] > R2)[0]
        for k in high:
            removed.add(window[k])
    del df_window
snp_ids = list(dict.fromkeys(pruned))
print(f"[QC] SNPs reduced to {len(snp_ids)} after missingness, MAF, LD pruning")

# Build genotype matrix DataFrame
geno_matrix = full_mat[[snp_index_map[s] for s in snp_ids], :]
geno_matrix = geno_matrix.T
df_geno = pd.DataFrame(
    geno_matrix,
    index=sample_ids,
    columns=snp_ids,
)
df_geno = df_geno.reset_index().rename(columns={"index": "GP2sampleID"})
df_geno_with_patno = pd.merge(
    df_geno,
    df_link[["GP2sampleID", "PATNO"]],
    on="GP2sampleID",
    how="inner"
)

# Merge genotype with clinical/cognitive
df_master = pd.merge(
    df_clin,
    df_geno_with_patno,
    on="PATNO",
    how="inner"
)
print(f"[2] Genotype merge → df_geno_with_patno has {len(df_geno_with_patno)} rows, df_master now has {len(df_master)} rows and {len(df_master.columns)} columns")

# ──────────────── 3) RNA-seq Data ────────────────
rna_keep_patnos = set(df_master["PATNO"].astype(str))
rna_folder = "/Volumes/Seagate BUP Slim SL Media/Reseach_Project/data_processing/raw/rna_seq"
sf_paths = glob.glob(os.path.join(rna_folder, "*.salmon.genes.sf"))
fail_folder = os.path.join(rna_folder, "fail")
if not sf_paths:
    raise FileNotFoundError(f"No ‘.salmon.genes.sf’ files found in {rna_folder}")
tpm_dfs = []
sample_ids = []
for sf_path in sf_paths:
    if fail_folder in sf_path:
        continue
    basename = os.path.basename(sf_path)
    if "BL" not in basename:
        continue
    # Extract PATNO from filename
    m = re.search(r"(IR3\.\d+)", basename)
    if not m:
        raise ValueError(f"Cannot extract sample ID from filename: {basename}")
    sample_with_prefix = m.group(1)
    patno = sample_with_prefix.replace("IR3.", "")
    sample_ids.append(patno)
    if patno not in rna_keep_patnos:
        continue
    df_sf = pd.read_csv(
        sf_path,
        sep="\t",
        usecols=["Name", "TPM"],
        index_col="Name",
        dtype={"TPM": float},
    )
    df_sf = df_sf.rename(columns={"TPM": patno})
    tpm_dfs.append(df_sf)
if not tpm_dfs:
    raise RuntimeError("No RNA .genes.sf files matched the PATNOs in df_master.")
# Ensure all gene indices match
canonical_genes = tpm_dfs[0].index.tolist()
for df in tpm_dfs[1:]:
    if not df.index.equals(pd.Index(canonical_genes)):
        raise ValueError("Gene order/mismatch detected between Salmon .genes.sf files!")
df_genes_by_sample = pd.concat(tpm_dfs, axis=1)
df_tpm = (
    df_genes_by_sample
    .transpose()
    .reset_index()
    .rename(columns={"index": "PATNO"})
)
print(f"[3] RNA-seq processed → {len(df_tpm)} samples, {len(df_tpm.columns) - 1} genes")

# Merge RNA-seq with master table
df_master = pd.merge(
    df_master,
    df_tpm,
    on="PATNO",
    how="inner"
)
print(f"[4] Master table final → {len(df_master)} rows, {len(df_master.columns)} columns")

# Feature column selection
clinical_cols = [
    "PATNO",
    "age_at_visit",
    "SEX_M",
    "EDUCYRS",
]
cognitive_cols_all = [
    "label"
]
transcriptomics_cols = [col for col in df_tpm.columns if col != "PATNO"]
genotype_cols = [
    col
    for col in df_geno_with_patno.columns
    if col not in ("GP2sampleID", "PATNO")
]

# ──────────────── Stratified CV Folds and Output ────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
output_dir = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/data_splits/cv_folds"
os.makedirs(output_dir, exist_ok=True)
for fold, (train_idx, test_idx) in enumerate(skf.split(df_master, df_master["label"])):
    df_train = df_master.iloc[train_idx].reset_index(drop=True)
    df_test = df_master.iloc[test_idx].reset_index(drop=True)
    # Univariate SNP filtering (FDR) on training fold
    fdr_vals = {}
    for snp in genotype_cols:
        values = df_train[snp]
        mask = values.notna()
        x = values[mask]
        y = df_train["label"][mask]
        if y.nunique() < 2 or x.shape[0] < 3:
            fdr_vals[snp] = 0.0
            continue
        x0 = x[y == 0]
        x1 = x[y == 1]
        if len(x0) < 2 or len(x1) < 2:
            fdr_vals[snp] = 0.0
            continue
        mu0, mu1 = x0.mean(), x1.mean()
        var0, var1 = x0.var(ddof=0), x1.var(ddof=0)
        fdr = (mu1 - mu0) ** 2 / (var1 + var0 + 1e-8)
        fdr_vals[snp] = fdr
    top_snps = sorted(fdr_vals, key=fdr_vals.get, reverse=True)[:1000]
    kept_columns = clinical_cols + cognitive_cols_all + transcriptomics_cols + top_snps
    df_train_filtered = df_train[kept_columns]
    df_test_filtered = df_test[kept_columns]
    train_path = os.path.join(output_dir, f"train_fold_{fold}_thresh_{args.threshold}.csv")
    test_path = os.path.join(output_dir, f"test_fold_{fold}_thresh_{args.threshold}.csv")
    df_train_filtered.to_csv(train_path, index=False)
    df_test_filtered.to_csv(test_path, index=False)
    print(f"[✓] Saved fold {fold} →")
    print(f"    {train_path}")
    print(f"    {test_path}")
