import pandas as pd

curated_path = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/raw/clinical_and_cognitive_curated/PPMI_Curated_Data_Cut_Public_20250321.xlsx"

curated = pd.read_excel(curated_path, sheet_name=0)

print(curated.shape)

for col in curated.columns:
    print(" -", col)