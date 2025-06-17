import pandas as pd
from sklearn.model_selection import train_test_split

# 1) Load your fully-preprocessed DataFrame (QC’d but not scaled or log-transformed)
df = pd.read_csv('/proposal_data/datasets_processed/rna_plus_clinical_final.csv')  # contains PATNO, clinical, RNA, moca_change

# 2) Bin the continuous target into four semantic categories for stratified split
bins = [-float('inf'), -3, -1, 0, float('inf')]
labels = ['large', 'moderate', 'slight', 'no_change']
df['y_bin'] = pd.cut(df['moca_change'], bins=bins, labels=labels)

# 3) Split off 10% as a held-out test set
df_train, df_test = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True,
    stratify=df['y_bin']  # remove this line if you prefer a purely random split
)

# 4) Drop the helper column
for split in (df_train, df_test):
    split.drop(columns=['y_bin'], inplace=True)

# 5) Save to disk
df_train.to_csv('/Users/nickq/Documents/Pioneer Academics/Research_Project/proposal_data/data_splits/initital_split/train_data.csv', index=False)
df_test.to_csv('/Users/nickq/Documents/Pioneer Academics/Research_Project/proposal_data/data_splits/initital_split/test_data.csv',  index=False)

print(f"Saved train ({len(df_train)}) and test ({len(df_test)}) splits.")