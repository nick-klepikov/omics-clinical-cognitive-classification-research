import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="Merge each omics with clinical")
parser.add_argument('--threshold', type=int, choices=[-2, -3, -4, -5],  required=True, help='Threshold for binarizing MoCA change')
args = parser.parse_args()


for fold in range(5):
    df_master = pd.read_csv(f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/data_splits/cv_folds/train_fold_{fold}_thresh_{args.threshold}.csv")

    clinical_cols = [
        "PATNO",
        "age_at_visit",
        "SEX_M",
        "EDUCYRS",
    ]

    transcriptomics_cols = [col for col in df_master.columns if col.startswith("ENSG")]
    genotype_cols = [
        col
        for col in df_master.columns
        if col not in clinical_cols + ["label"] + transcriptomics_cols
    ]
    # genotype + clinical
    df_geno_clin = df_master[clinical_cols + ["label"] + genotype_cols]
    df_geno_clin.to_csv(
        f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/final_datasets_unprocessed/geno_plus_clinical_fold_{fold}_thresh_{args.threshold}.csv",
        index=False
    )
    # transcriptomics + clinical
    df_rna_clin = df_master[clinical_cols + ["label"] + transcriptomics_cols]
    df_rna_clin.to_csv(
        f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/final_datasets_unprocessed/rna_plus_clinical_fold_{fold}_thresh_{args.threshold}.csv",
        index=False
    )