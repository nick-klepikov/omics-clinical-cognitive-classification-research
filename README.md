# Classification Pipeline

This repository provides three main сomponents for an end-to-end pipeline:
1. **Pre-processing**: prepare feature matrices and affinity matrices
2. **Graph-Based GCN Training**: training and hyperparameter tuning of a Graph Convolutional Network, additionally training, tuning, and evaluating RandomForest, SVM, and XGBoost classifiers
3. **Model Testing**: evaluate GCN on a test data split

---

## 1. Pre-processing Pipeline

**Folder**  
`src/classification_pipeline/data_processing`

**Description**  
Perform quality control, feature selection, and fold-wise data assembly.  
Outputs per-fold mastertable CSVs and affinity matrices (`.npy`), combining all modalities.

**Usage**  
```bash
python preprocessing_pipeline.py
  --threshold         -5
```

**Flags**
- `--threshold -5`
      MoCA-change cutoff for labeling decline

**Options**
- `--threshold` can be one of: `-2`, `-3`, `-4`, `-5` (choose the MoCA-change cutoff for decline labeling)

---

## 2. Graph-Based GCN Training

**Folder**  
`src/classification_pipeline/model_training`

**Description**  
Trains a 2-layer GCN and performs nested five-fold nested cross-validation for hyperparameter and threshold tuning. Exports the best set of hyperparams for a current configu.

This folder also includes `dos_gnn_model_runner.py`, an implementation of DOS-GNN from Jing et al., “DOS-GNN: Dual-Feature Aggregations with Over-Sampling for Class-Imbalanced Fraud Detection on Graphs” (IJCNN 2024, DOI:10.1109/IJCNN60899.2024.10650494), which uses the same flags as above and has the same output.

**Usage**  
```bash
python graphsmote_runner.py \
  --config     configs/config_ablation_threshold_-4.yaml \
  --out_dir    data/results/ablation_experiments_2      \
  --mastertable data/intermid/fused_datasets/final_mastertable \
  --modality   rna                                                  \
  --model      GCNN                                                 \
  --threshold  -4
```

**Flags**
- `--config "src/classification_pipeline/configs/config_ablation_threshold_-4.yaml"`
      Hyperparameter YAML file
- `--out_dir "data/results/ablation_experiments_2"`
      Output directory for GCN results
- `--mastertable "data/intermid/fused_datasets/final_mastertable"`
      Prefix for per-fold feature matrices filenames; actual files named `<PREFIX>_fold_{i}_thresh_{t}.csv`
- `--modality "fused"`
      Feature modality
- `--model "GCNN"`
      Model name
- `--threshold -4`
      MoCA-change cutoff

**Options**
- `--modality`: `geno`, `rna`, or `fused`
- `--model`: `GCNN`, `MLP2`, or DOS_GNN`
- `--threshold`: `-2`, `-3`, `-4`, or `-5`

---

## 3. Traditional ML Training

**Script**  
`src/classification_pipeline/model_runner.py`

**Description**  
Runs nested grid-search CV on RandomForest, SVM, and XGBoost (with optional SMOTE/BorderlineSMOTE), and saves per-fold performance metrics and ROC inputs, plus final summary.

**Usage**  
```bash
python model_runner.py \
  --out_dir     data/results/ablation_experiments \
  --mastertable data/intermid/fused_datasets/final_mastertable \
  --modality    fused                                             \
  --threshold   -4
```

**Flags**
- `--out_dir "data/results/ablation_experiments"`
      Output directory for ML results
- `--mastertable "data/intermid/fused_datasets/final_mastertable"`
      Prefix for per-fold feature matrices filenames; actual files named `<PREFIX>_fold_{i}_thresh_{t}.csv`
- `--modality "fused"`
      Feature modality
- `--threshold -4`
      MoCA-change cutoff

**Options**
- `--modality`: `geno`, `rna`, or `fused`
- `--threshold`: `-2`, `-3`, `-4`, and `-5`

## 4. Model Testing

**Folder**  
`src/classification_pipeline/model_testing`

**Description**  
Evaluate trained models on the test split and generate ROC plots. Uses the same flags as the corresponding training scripts. This folder also includes `graphsmote_runner.py`, implementing GraphSMOTE from Zhang et al., “GraphSMOTE: Graph-based synthetic minority over-sampling technique for imbalanced node classification” (KDD 2021, DOI:10.1145/3447548.3467435), and it uses the same configuration files as the training scripts.

---

## Repository Structure

```
./
Research_Project/
├── data/
│   ├── results/                  ← analysis outputs (tuning, ablations, comparisons…)
│   ├── figures/                  ← assets 1–7 (plots, tables)
│   ├── intermid/                 ← intermediate datasets & affinity matrices
│   └── raw/                      ← raw clinical, RNA-seq & genetics data
└── src/
    ├── classification_pipeline/
    │   ├── post_analysis/        ← stats & DAVID enrichment results
    │   ├── data_processing/      ← ETL & feature-matrix pipelines
    │   ├── utils/                ← helper functions
    │   ├── models/               ← model definitions (GCNN, DOS-GNN…)
    │   ├── configs/              ← YAML hyperparam & ablation configs
    │   ├── model_training/       ← training & tuning runners
    │   └── model_testing/        ← evaluation & test scripts
```

---

## Pipeline Workflow

1. **Pre-processing**  
   Run each script in `src/classification_pipeline/data_processing` in sequence. These scripts:
   - Load raw clinical, genetics, and RNA-seq inputs.
   - Apply QC, feature selection, and per-fold splits.
   - Output mastertables (feature matrices) (`.csv`) and affinity matrices (`.npy`) to `data/intermid/...`.

2. **Model Training**  
   - **Graph-Based GCN & DOS-GNN**  
     Run the training runners in `src/classification_pipeline/model_training` (e.g. `model_runner.py`, `dos_gnn_model_runner.py`). They read the pre-processed mastertables and write hyperparameter-tuning outputs to `data/results/...`.  
   - **Traditional ML**  
     Invoke `src/classification_pipeline/trad_ml_runner_and_eval.py` to train Random Forest, SVM, and XGBoost via nested CV. Outputs (metrics, ROC inputs, summary) appear in `data/results/...`.

3. **Model Testing**  
   Execute the evaluation scripts in `src/classification_pipeline/model_testing` on the held-out test fold. They use trained models and mastertables to generate final ROC plots and metric tables under `data/results/...`.
