import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# === CONFIGURE PATHS ===
INPUT_CSV  = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_unprocessed/cognitive_plus_clinical.csv"
OUTPUT_CSV = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_datasets_preprocessed/cognitive_plus_clinical_final.csv"

# 1) LOAD DATA
df = pd.read_csv(INPUT_CSV, dtype={"PATNO": str})

# 2) DROP UNUSED COLUMNS
df = df.drop(columns=["EVENT_ID", "subgroup", "race"], errors="ignore")

 # SEX is already coded as 1=male, 0=female, so copy directly
df["SEX_M"] = df["SEX"].astype(int)
df = df.drop(columns=["SEX"], errors="ignore")

# 4) IDENTIFY CONTINUOUS AND BINARY COLUMNS
cont_cols = [
    "age_at_visit",   # age in years
    "EDUCYRS",        # years education
    "moca",           # baseline MoCA
    "moca_12m"        # 12-month MoCA
]
bin_cols = ["SEX_M"]  # binary sex indicator, keep as 0/1

# 5) MEAN-IMPUTE MISSING VALUES
imputer = SimpleImputer(strategy="mean")
df[cont_cols] = imputer.fit_transform(df[cont_cols])
df[bin_cols]  = imputer.fit_transform(df[bin_cols])

# 6) Z-SCORE NORMALIZE CONTINUOUS FEATURES ONLY
scaler = StandardScaler()
df[cont_cols] = scaler.fit_transform(df[cont_cols])
# bin_cols remain as 0/1

# 7) SAVE PREPROCESSED CSV
cols = ["PATNO"] + cont_cols + bin_cols
df[cols].to_csv(OUTPUT_CSV, index=False)
print(f"Wrote preprocessed clinical+cognitive data to {OUTPUT_CSV}")