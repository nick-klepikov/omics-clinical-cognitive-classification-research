import pandas as pd  # pandas: library for data manipulation and analysis
import re  # re: Python’s regular‐expression module, used for pattern matching
from pandas_plink import read_plink
import numpy as np
import glob
import os
from scipy.stats import pearsonr

# ────────────────────────────────────────────────────────────────────────────────
# 1) Load Clinical & Cognitive data
# ────────────────────────────────────────────────────────────────────────────────

# Path to the Excel file containing clinical & cognitive information.
clinical_path = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/raw/clinical_and_cognitive_curated/PPMI_Curated_Data_Cut_Public_20250321.xlsx"

# Read the “Baseline” sheet from the Excel file into a pandas DataFrame.
# We assume there is a column named "PATNO".
df_clin = pd.read_excel(
    clinical_path,
    sheet_name="20250310",
    dtype={"PATNO": str}
)
#   • pd.read_excel: loads an Excel sheet into a DataFrame.
#   • sheet_name="20250310": selects only the “20250310” sheet.
#   • dtype={"PATNO": str}: ensure PATNO is read as a string.

# Original subgroup-based filtering for clinical/cognitive data
if "EVENT_ID" in df_clin.columns:
    df_clin_bl = df_clin[(df_clin["EVENT_ID"] == "BL") & (df_clin["subgroup"] == "Sporadic PD")].copy()
    df_v04 = df_clin[(df_clin["EVENT_ID"] == "V04") & (df_clin["subgroup"] == "Sporadic PD")].copy()

    # Rename only the "moca" column in df_v04 to "moca_12m"
    if "moca" in df_v04.columns:
        df_v04 = df_v04.rename(columns={"moca": "moca_12m"})

    # Keep PATNO and moca_12m in the 12-month DataFrame
    v04_keep = ["PATNO"]
    if "moca_12m" in df_v04.columns:
        v04_keep.append("moca_12m")
    df_v04 = df_v04[v04_keep]

    # Merge baseline and 12-month cognitive on PATNO
    df_clin = pd.merge(
        df_clin_bl,
        df_v04,
        on="PATNO",
        how="left"
    )
    # Drop any subjects missing either baseline or 12m MoCA
    df_clin = df_clin.dropna(subset=["moca", "moca_12m"]).reset_index(drop=True)
    df_clin["moca"] = pd.to_numeric(df_clin["moca"], errors="coerce")
    df_clin["moca_12m"] = pd.to_numeric(df_clin["moca_12m"], errors="coerce")
    df_clin["moca_change"] = df_clin["moca_12m"] - df_clin["moca"]
# Decide which clinical columns to keep. We want PATNO plus any columns we need:
keep_cols = ["PATNO", "age_at_visit", "SEX", "EDUCYRS", "moca_change"]
# Filter df_clin to include only those columns that actually exist in the DataFrame.
df_clin = df_clin[[c for c in keep_cols if c in df_clin.columns]].copy()
#   • This list comprehension chooses only keep_cols that exist, avoiding KeyError.
print(f"[1] Clinical → {len(df_clin)} rows, {len(df_clin.columns)} columns after filtering")
clinic_patnos = set(df_clin["PATNO"])


# ────────────────────────────────────────────────────────────────────────────────
# 2) Load Genotyping
# ────────────────────────────────────────────────────────────────────────────────

# Load the IID→PATNO linklist.
df_link = pd.read_csv(
    "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/raw/genetics/PPMI_244_Plink/ppmi_244_linklist.csv",
    dtype={"GP2sampleID": str, "PATNO": str, "COHORT": str})
clinic_iids = set(
    df_link.loc[df_link["PATNO"].isin(clinic_patnos), "GP2sampleID"]
)


# 1. Point “read_plink” at the common prefix (no extension) of your .bed/.bim/.fam trio.
#    For example, if your files are:we
#        ├─ genotypes.bed
#        ├─ genotypes.bim
#        └─ genotypes.fam
#
#    Then you pass read_plink("genotypes").
#    It will look for “genotypes.bed”, etc.

(bim, fam, bed) = read_plink("/Users/nickq/Documents/Pioneer Academics/Research_Project/data/raw/genetics/PPMI_244_Plink/nonGR_LONI_PPMI_MAY2023")
#   • bim is a pandas-plonk GenotypeDataFrame for the SNP annotation (.bim)
#   • fam is a pandas-plonk FamilyDataFrame for the sample info (.fam)
#   • bed is a dask array of shape (n_variants, n_samples) containing 0/1/2 (allele counts)

# Create a boolean mask of which samples (rows in fam) to keep: only those IIDs in clinic_iids
# Subset fam and bed to only those samples
keep_samples = fam["iid"].isin(clinic_iids).values
fam = fam[keep_samples].reset_index(drop=True)
bed = bed[:, keep_samples]

# --- Compute full genotype matrix once to avoid repeated Dask I/O ---
full_mat = bed.compute()  # NumPy array shape = (n_variants, n_samples)

#    a) Grab sample IDs in order:
sample_ids = fam.iid.values.tolist()

#    b) Grab SNP IDs (one column per “snp” in bim):
snp_ids = bim.snp.values.tolist()

# ----------------- SNP QC before loading full genotype matrix -----------------

# Map snp_ids to their indices in the original bim.snp list
snp_index_map = {s: idx for idx, s in enumerate(bim.snp.values.tolist())}

# 1) Batch missingness & MAF filter
batch_size = 10000
good_snps = []
for start in range(0, len(snp_ids), batch_size):
    print(f"[QC: missing+MAF] Batch {start}-{min(start+batch_size, len(snp_ids))} of {len(snp_ids)} SNPs")
    batch = snp_ids[start:start+batch_size]
    # load batch of variants x all samples
    arr = full_mat[[snp_index_map[s] for s in batch], :].T
    df_batch = pd.DataFrame(arr, columns=batch)
    # missingness
    miss = df_batch.isna().mean()
    keep1 = miss[miss <= 0.05].index.tolist()
    # MAF
    p = df_batch[keep1].mean(skipna=True) / 2
    maf = np.minimum(p, 1-p)
    keep2 = maf[maf >= 0.01].index.tolist()
    good_snps.extend(keep2)
    del df_batch

# dedupe order
good_snps = list(dict.fromkeys(good_snps))


# 2) LD pruning on good_snps
pruned = []
WINDOW, STEP, R2 = 50, 5, 0.2
for i in range(0, len(good_snps), STEP):
    print(f"[QC: LD prune] Window start {i} of {len(good_snps)} SNPs")
    window = good_snps[i:i+WINDOW]
    arr = full_mat[[snp_index_map[s] for s in window], :].T
    df_window = pd.DataFrame(arr, columns=window)
    col_means = df_window.mean()
    df_window = df_window.fillna(col_means)

    # Skip pruning when only one SNP in window
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

# Final pruned SNP list
snp_ids = list(dict.fromkeys(pruned))
print(f"[QC] SNPs reduced to {len(snp_ids)} after missingness, MAF, LD pruning")

#    c) Compute the full genotype matrix in memory (if you have RAM):
geno_matrix = full_mat[[snp_index_map[s] for s in snp_ids], :]  # shape = (len(snp_ids), n_samples)

#    d) Transpose so that rows become samples, columns become variants:
geno_matrix = geno_matrix.T  # now shape = (n_samples, n_variants)

#    e) Build a pandas DataFrame:
df_geno = pd.DataFrame(
    geno_matrix,
    index=sample_ids,
    columns=snp_ids,
)

df_geno = df_geno.reset_index().rename(columns={"index": "GP2sampleID"})
#    Now df_geno has columns: ["IID", <all SNP IDs>],
#    and one row per IID.

df_geno_with_patno = pd.merge(
    df_geno,      # left: genotype calls with an "IID" column
    df_link[["GP2sampleID", "PATNO"]],  # right: mapping of IID→PATNO
    on="GP2sampleID",
    how="inner"         # keep only IIDs that appear in both df_geno_reset and df_link
)
#    Resulting columns: ["IID", <all SNPs>, "PATNO"]

# Merge that (which now has a "PATNO" column) onto df_clin, keyed by PATNO
df_master = pd.merge(
    df_clin,               # clinical/cognitive (keyed by "PATNO")
    df_geno_with_patno,    # genotype calls (keyed by "PATNO" after merge)
    on="PATNO",
    how="inner"            # keep only samples present in both modalities
)
print(f"[2] Genotype merge → df_geno_with_patno has {len(df_geno_with_patno)} rows, df_master now has {len(df_master)} rows and {len(df_master.columns)} columns")

# ────────────────────────────────────────────────────────────────────────────────
# 3) Load RNA-seq
# ────────────────────────────────────────────────────────────────────────────────

rna_keep_patnos = set(df_master["PATNO"].astype(str))

# 1) Path to your folder of Salmon “.genes.sf” files
rna_folder = "/Volumes/Seagate BUP Slim SL Media/Reseach_Project/data/raw/rna_seq"

# 2) Find all files ending with “.salmon.genes.sf”
sf_paths = glob.glob(os.path.join(rna_folder, "*.salmon.genes.sf"))
# Exclude QC‑failed samples in the “fail” subdirectory
fail_folder = os.path.join(rna_folder, "fail")
if not sf_paths:
    raise FileNotFoundError(f"No ‘.salmon.genes.sf’ files found in {rna_folder}")

# 3) Loop over each .genes.sf file, read “Name” and “TPM”, storing them in a list
tpm_dfs = []
sample_ids = []

for sf_path in sf_paths:
    # skip QC‑failed samples
    if fail_folder in sf_path:
        continue
    basename = os.path.basename(sf_path)
    # Only process baseline files (those containing "BL" in the filename)
    if "BL" not in basename:
        continue
    # ----------------------------------------------------------------------------
    # Extract “IR3.3000” using regex.
    # We look for the pattern “IR3.<one or more digits>” anywhere in the filename.
    m = re.search(r"(IR3\.\d+)", basename)
    if not m:
        # If it doesn’t match, you can choose to skip or raise an error:
        raise ValueError(f"Cannot extract sample ID from filename: {basename}")
    sample_with_prefix = m.group(1)     # e.g. "IR3.3000"
    # If you want “3000” (to match PATNO in your clinical/genotype), strip the “IR3.”:
    patno = sample_with_prefix.replace("IR3.", "")  # → "3000"
    sample_ids.append(patno)

    if patno not in rna_keep_patnos:
        continue

    # 3a) Read only the “Name” (gene ID) and “TPM” columns, indexing by gene name
    df_sf = pd.read_csv(
        sf_path,
        sep="\t",
        usecols=["Name", "TPM"],
        index_col="Name",
        dtype={"TPM": float},
    )
    # Rename the TPM column to the stripped PATNO (so columns align later)
    df_sf = df_sf.rename(columns={"TPM": patno})

    # 3b) Append to our list
    tpm_dfs.append(df_sf)

if not tpm_dfs:
    raise RuntimeError("No RNA .genes.sf files matched the PATNOs in df_master.")

# 4) Verify that all DataFrames share the same index (gene order).
canonical_genes = tpm_dfs[0].index.tolist()
for df in tpm_dfs[1:]:
    if not df.index.equals(pd.Index(canonical_genes)):
        raise ValueError("Gene order/mismatch detected between Salmon .genes.sf files!")

# 5) Concatenate all small DataFrames into one big “genes × samples” matrix
df_genes_by_sample = pd.concat(tpm_dfs, axis=1)
# Now df_genes_by_sample.index = [ENSG IDs], columns = ["3000", "3001", “3002”, …]

# 6) Transpose so that rows are samples (PATNO) and columns are genes
df_tpm = (
    df_genes_by_sample
    .transpose()
    .reset_index()
    .rename(columns={"index": "PATNO"})
)
print(f"[3] RNA-seq processed → {len(df_tpm)} samples, {len(df_tpm.columns) - 1} genes")
# Now df_tpm has columns: ["PATNO", "ENSG000001xxxxxxxx", "ENSG000002yyyyyyyy", …]

# 4.9) Final merge: inner‐join df_master with df_tpm on “PATNO”
df_master = pd.merge(
    df_master,   # clinical + genotype (keyed by PATNO)
    df_tpm,      # RNA‐TPM matrix (only for PATNOs that overlap)
    on="PATNO",
    how="inner"  # keep only samples present in BOTH
)
print(f"[4] Master table final → {len(df_master)} rows, {len(df_master.columns)} columns")


# 1) Define which clinical columns to include
clinical_cols = [
    "PATNO",
    "age_at_visit",
    "SEX",
    "EDUCYRS",
]

# 2) Define cognitive‐score columns (only those that actually exist in df_master)
cognitive_cols_all = [
    "moca_change"
]

# 3) Define transcriptomics columns by taking everything in df_tpm except “PATNO”
transcriptomics_cols = [col for col in df_tpm.columns if col != "PATNO"]

# 4) Define genotype‐only columns by taking everything in df_geno_with_patno except:
#       – "GP2sampleID"  (we don’t need the raw IID in the final tables)
#       – "PATNO"       (already in clinical_cols)
# If there are any other genetic flags you prefer to keep separate, exclude them here as well.
genotype_cols = [
    col
    for col in df_geno_with_patno.columns
    if col not in ("GP2sampleID", "PATNO")
]

# --- Univariate SNP filtering by correlation with moca_change ---
corr_vals = {}
for snp in genotype_cols:
    mask = df_master[snp].notna() & df_master["moca_change"].notna()
    if mask.sum() > 2:
        r, _ = pearsonr(df_master.loc[mask, snp], df_master.loc[mask, "moca_change"])
        corr_vals[snp] = abs(r)
    else:
        corr_vals[snp] = 0.0
# Select top 1000 by |r|
top_snps = sorted(corr_vals, key=corr_vals.get, reverse=True)[:1000]
genotype_cols = top_snps
print(f"[QS] Univariate SNP filter → retained {len(genotype_cols)} SNPs (top 1000 by |r|)")


# ────────────────────────────────────────────────────────────────────────────────
# 5) Build and save: GENOTYPE + CLINICAL
# ────────────────────────────────────────────────────────────────────────────────
df_geno_clin = df_master[clinical_cols + ["moca_change"] + genotype_cols]
print("[DEBUG] About to write geno_plus_clinical.csv with shape:", df_geno_clin.shape)
df_geno_clin.to_csv(
    "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_unprocessed/geno_plus_clinical.csv",
    index=False
)
print(f"[5] Written geno_plus_clinical.csv ({df_geno_clin.shape[0]} samples)")

# ────────────────────────────────────────────────────────────────────────────────
# 6) Build and save: TRANSCRIPTOMICS + CLINICAL
# ────────────────────────────────────────────────────────────────────────────────
df_rna_clin = df_master[clinical_cols + ["moca_change"] + transcriptomics_cols]
df_rna_clin.to_csv(
    "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_unprocessed/rna_plus_clinical.csv",
    index=False
)
print(f"[6] Written rna_plus_clinical.csv ({df_rna_clin.shape[0]} samples)")

print("Wrote two CSVs to data/processed/:")
print(f"  • geno_plus_clinical.csv: {df_geno_clin.shape[0]} samples")
print(f"  • rna_plus_clinicale.csv: {df_rna_clin.shape[0]} samples")
