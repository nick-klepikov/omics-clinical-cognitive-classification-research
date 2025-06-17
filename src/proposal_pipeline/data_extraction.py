# rna_cognitive_extraction.py

import pandas as pd
import glob
import os
import re

# 1) Load Clinical & Cognitive data
df_clin = pd.read_excel(
    "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/raw/clinical_and_cognitive_curated/PPMI_Curated_Data_Cut_Public_20250321.xlsx",
    sheet_name="20250310",
    dtype={"PATNO": str}
)
if "EVENT_ID" in df_clin:
    bl = df_clin[(df_clin.EVENT_ID == "BL") & (df_clin.subgroup == "Sporadic PD")]
    v04 = df_clin[(df_clin.EVENT_ID == "V04") & (df_clin.subgroup == "Sporadic PD")].rename(columns={"moca": "moca_12m"})
    v04 = v04[["PATNO", "moca_12m"]]
    df_clin = bl.merge(v04, on="PATNO", how="inner")
    # Ensure both baseline and 12-month MoCA scores and sex are present
    df_clin = df_clin.dropna(subset=["moca", "moca_12m", "SEX"])
keep = [c for c in ["PATNO","age_at_visit","SEX","EDUCYRS", "moca","moca_12m"] if c in df_clin]
df_clin = df_clin[keep]

df_clin["moca_change"] = df_clin["moca_12m"] - df_clin["moca"]
clinical_cols = df_clin.columns.tolist()


# 2) Load RNA-seq
rna_folder = "/Volumes/Seagate BUP Slim SL Media/Reseach_Project/data/raw/rna_seq"
sf_paths = glob.glob(os.path.join(rna_folder, "*.salmon.genes.sf"))
tpm_dfs = []
for path in sf_paths:
    if "fail" in path or "BL" not in path: continue
    m = re.search(r"IR3\.(\d+)", os.path.basename(path))
    if not m: continue
    patno = m.group(1)
    if patno not in df_clin.PATNO.values: continue
    df = pd.read_csv(path, sep="\t", usecols=["Name","TPM"], index_col="Name").rename(columns={"TPM": patno})
    tpm_dfs.append(df)
if not tpm_dfs:
    raise RuntimeError("No matching RNA files")
genes = tpm_dfs[0].index
for df in tpm_dfs[1:]:
    if not df.index.equals(genes):
        raise ValueError("Gene index mismatch")
df_tpm = pd.concat(tpm_dfs, axis=1).T.reset_index().rename(columns={"index": "PATNO"})

# Merge clinical and RNA-seq into df_master
df_master = pd.merge(df_clin, df_tpm, on="PATNO", how="inner")

# 3) Define columns
transcriptomics_cols = [c for c in df_tpm.columns if c != "PATNO"]

# 4) Export tables
df_master[clinical_cols + transcriptomics_cols] \
    .to_csv("/Users/nickq/Documents/Pioneer Academics/Research_Project/proposal_data/datasets_unprocessed/rna_plus_clinical.csv", index=False)
# Final count of samples
print(f"Total samples for modeling: {len(df_master)}")