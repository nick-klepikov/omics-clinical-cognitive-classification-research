from src.classification_pipeline.utils.utils import *
import yaml
import os
import argparse
import matplotlib
import numpy as np
import pandas as pd
import time
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import numpy as np

# --- Threshold tuning utility ---
def find_best_threshold(probs, labels, metric=f1_score):
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_thresh, best_score = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = metric(labels, preds)
        if score > best_score:
            best_score, best_thresh = score, t
    return best_thresh, best_score
import os
os.environ["OMP_NUM_THREADS"] = "1"

start_time = time.time()

# ------------ Argument parsing ------------
parser = argparse.ArgumentParser(description="Train GCN on multi-omics data_processing")
parser.add_argument('--out_dir',     type=str, required=True, help='Directory for outputs')
parser.add_argument('--mastertable', type=str, required=True, help='Path to mastertable CSV')
parser.add_argument('--modality',   type=str, choices=["geno", "rna", "fused"], default="fused", help='Modality to train on')
parser.add_argument('--threshold', type=int, choices=[-2, -3, -4],  required=True, help='Threshold for binarizing MoCA change')
args = parser.parse_args()

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

device = check_cuda()

# Set matplotlib to 'agg'
if matplotlib.get_backend() != 'agg':
    matplotlib.use('agg')

#----------------  Main function -----------------#
if __name__ == '__main__':
    # Define param grids for each model (with clf__ prefix for pipeline)
    param_grid = {
        "RandomForest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [5, 10, None],
            "clf__random_state": [42]
        },
        "SVM": {
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["linear", "rbf"],
            "clf__probability": [True]
        },
        "XGBoost": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [5, 7],
            "clf__learning_rate": [0.05, 0.1],
            "clf__random_state": [42],
            "clf__subsample": [1.0],
            "clf__gamma": [0]
        }
    }
    # Each model is a pipeline: SMOTE oversampling + classifier
    models = {
        "RandomForest": lambda: Pipeline([
            ("smote", BorderlineSMOTE(random_state=42)),
            ("clf", RandomForestClassifier())
        ]),
        "SVM": lambda: Pipeline([
            ("smote", BorderlineSMOTE(random_state=42)),
            ("clf", SVC())
        ]),
        "XGBoost": lambda: Pipeline([
            ("smote", BorderlineSMOTE(random_state=42)),
            ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=1))
        ])
    }

    # Outer results dict
    nested_cv_results = {}

    for outer_fold in range(5):
        print(f"Starting outer fold {outer_fold}")
        mastertable_file = f"{args.mastertable}_fold_{outer_fold}_thresh_{args.threshold}.csv"
        mastertable = pd.read_csv(mastertable_file, index_col="PATNO")
        # 1) Modality filter (exactly how you did for the GNN branch)
        if args.modality == "geno":
            feat_df = mastertable[[c for c in mastertable.columns if not c.startswith("ENSG")]]
        elif args.modality == "rna":
            feat_df = mastertable[[c for c in mastertable.columns
                                   if c.startswith("ENSG")
                                   or c in ["label", "split", "age_at_visit", "SEX_M", "EDUCATION_YEARS"]]]
        else:
            feat_df = mastertable.drop(columns=["label", "split"])
        print(f"[Fold {outer_fold}] Features shape: {feat_df.shape}")
        split = mastertable["split"]
        trainval_mask = (split != "test").to_numpy()
        test_mask = (split == "test").to_numpy()
        X_all = feat_df.to_numpy()
        y_all = mastertable["label"].astype(int).to_numpy()
        X_tv, y_tv = X_all[trainval_mask], y_all[trainval_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        print(f"[Fold {outer_fold}] Train/val samples: {X_tv.shape[0]}, Test samples: {X_test.shape[0]}")

        for model_name, model_cls in models.items():
            print(f"[Fold {outer_fold}] Running nested CV for: {model_name}")
            # Inner CV: 5 folds on train+val
            grid = GridSearchCV(
                models[model_name](),
                param_grid=param_grid[model_name],
                scoring="f1",
                cv=5,
                refit=True,
                n_jobs=1
            )
            grid.fit(X_tv, y_tv)
            best_model = grid.best_estimator_

            # --- Threshold tuning on train+validation set ---
            tv_probs = best_model.predict_proba(X_tv)[:, 1]
            tv_labels = y_tv
            best_thresh, best_f1 = find_best_threshold(tv_probs, tv_labels)
            print(f"[Fold {outer_fold}][{model_name}] Tuned threshold: {best_thresh:.2f}, F1 on trainval: {best_f1:.3f}")

            # Apply tuned threshold to test set
            proba = best_model.predict_proba(X_test)[:, 1]
            preds = (proba >= best_thresh).astype(int)
            # Compute minority-class metrics
            auc = roc_auc_score(y_test, proba)
            precision = precision_score(y_test, preds, pos_label=1)
            recall = recall_score(y_test, preds, pos_label=1)
            f1_binary = f1_score(y_test, preds, average="binary", pos_label=1)

            # Debug: print class distributions
            print(f"y_true class distribution: {np.bincount(y_test)}")
            print(f"y_pred class distribution: {np.bincount(preds)}")

            # Debug: print confusion matrix
            print("Confusion Matrix for this fold:")
            print(confusion_matrix(y_test, preds))

            # Log minority-class performance
            print(f"[Fold {outer_fold}][{model_name}] Precision (minority): {precision:.4f}, Recall (minority): {recall:.4f}, F1 (minority): {f1_binary:.4f}, AUC: {auc:.4f}")

            # Store metrics
            fold_key = f"{model_name}_fold_{outer_fold}"
            nested_cv_results[fold_key] = {
                "auc": auc,
                "precision_minority": precision,
                "recall_minority": recall,
                "f1_minority": f1_binary,
                "best_config": grid.best_params_
            }

    # Save all results to JSON
    out_path = os.path.join(OUT_DIR, "traditional_ml_nested_gridsearch_results.json")
    with open(out_path, "w") as f:
        json.dump(nested_cv_results, f, indent=2)
    print(f"Saved nested CV results to {out_path}")

    # --------- Compute summary statistics and save to summary JSON ---------
    # Group results by model
    results = {}
    for fold_key, metrics in nested_cv_results.items():
        # fold_key is like "RandomForest_fold_0"
        model_name = fold_key.split("_fold_")[0]
        if model_name not in results:
            results[model_name] = []
        results[model_name].append(metrics)

    summary = {}
    for model_name, fold_metrics in results.items():
        all_f1 = [m["f1_minority"] for m in fold_metrics]
        all_auc = [m["auc"] for m in fold_metrics]
        all_precision = [m["precision_minority"] for m in fold_metrics]
        all_recall = [m["recall_minority"] for m in fold_metrics]
        summary[model_name] = {
            "f1_minority_mean": float(np.mean(all_f1)),
            "f1_minority_std": float(np.std(all_f1)),
            "precision_minority_mean": float(np.mean(all_precision)),
            "precision_minority_std": float(np.std(all_precision)),
            "recall_minority_mean": float(np.mean(all_recall)),
            "recall_minority_std": float(np.std(all_recall)),
            "auc_mean": float(np.mean(all_auc)),
            "auc_std": float(np.std(all_auc)),
        }

    summary_path = os.path.join(OUT_DIR, "traditional_ml_nested_gridsearch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved summary statistics to {summary_path}")
