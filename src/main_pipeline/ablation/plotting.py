import pandas as pd
import matplotlib.pyplot as plt
import ast

# 1) Load the CSV you showed
df = pd.read_csv("/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_res/ablation_experiments/features_track_fused_GCNN.csv")

# 2) Parse the 'Raw_Feature_Importances' dicts
#    Each cell is a string like "{'Importance_score': {feat: score, ...}}"
raw_dicts = df['Raw_Feature_Importances'].apply(lambda s: ast.literal_eval(s)['Importance_score'])

# 3) Build a DataFrame: rows=folds, cols=features, values=importance (NaN if missing)
imp_df = pd.DataFrame(raw_dicts.tolist())

# 4) Filter to features present in at least 2 folds
valid = imp_df.count(axis=0) >= 2
filtered = imp_df.loc[:, valid]

# 5) Compute the average importance and pick top 20
avg_imp = filtered.mean(axis=0).sort_values(ascending=False)
# Normalize to percentages summing to 100%
rel_imp = avg_imp / avg_imp.sum() * 100
top20 = rel_imp.head(20)

# 6) Plot
plt.figure(figsize=(8, 10))
bars = plt.barh(top20.index, top20.values)
plt.xlabel("Relative Importance (%)")
plt.title("Top 20 Consistent Features (≥2 folds)")

# Invert y-axis so highest-scoring at the top
plt.gca().invert_yaxis()

# Annotate bars with their numeric value (percentage)
for bar in bars:
    width = bar.get_width()
    plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
             f"{width:.1f}%", va='center')

plt.tight_layout()
plt.savefig("Fused_GCN_Top20_Features.png", dpi=300)
plt.show()