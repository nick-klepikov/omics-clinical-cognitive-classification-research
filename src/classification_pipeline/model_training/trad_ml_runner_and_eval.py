import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from src.classification_pipeline.utils.utils import check_cuda

def find_best_threshold(probs, labels, metric=f1_score):
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_thresh, best_score = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = metric(labels, preds)
        if score > best_score:
            best_score, best_thresh = score, t
    return best_thresh, best_score

os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser(description="Train GCN on multi-omics data_processing")
parser.add_argument('--out_dir',     type=str, required=True, help='Directory for outputs')
parser.add_argument('--mastertable', type=str, required=True, help='Path to mastertable CSV')
parser.add_argument('--modality',   type=str, choices=["geno", "rna", "fused"], default="fused", help='Modality to train on')
parser.add_argument('--threshold', type=int, choices=[-2, -3, -4],  required=True, help='Threshold for binarizing MoCA change')
args = parser.parse_args()

USE_SMOTE = True
USE_THRESHOLD_TUNING = True

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

device = check_cuda()

if matplotlib.get_backend() != 'agg':
    matplotlib.use('agg')

if __name__ == '__main__':
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
    from imblearn.over_sampling import BorderlineSMOTE

    def make_pipeline(clf):
        steps = []
        if USE_SMOTE:
            steps.append(("smote", BorderlineSMOTE(random_state=42)))
        steps.append(("clf", clf))
        return Pipeline(steps)

    models = {
        "RandomForest": lambda: make_pipeline(RandomForestClassifier()),
        "SVM": lambda: make_pipeline(SVC()),
        "XGBoost": lambda: make_pipeline(XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=1))
    }

    nested_cv_results = {}

    for outer_fold in range(5):
        mastertable_file = f"{args.mastertable}_fold_{outer_fold}_thresh_{args.threshold}.csv"
        mastertable = pd.read_csv(mastertable_file, index_col="PATNO")
        if args.modality == "geno":
            feat_df = mastertable[[c for c in mastertable.columns if not c.startswith("ENSG")]]
        elif args.modality == "rna":
            feat_df = mastertable[[c for c in mastertable.columns
                                   if c.startswith("ENSG")
                                   or c in ["label", "split", "age_at_visit", "SEX_M", "EDUCATION_YEARS"]]]
        else:
            feat_df = mastertable.drop(columns=["label", "split"])
        split = mastertable["split"]
        trainval_mask = (split != "test").to_numpy()
        test_mask = (split == "test").to_numpy()
        X_all = feat_df.to_numpy()
        y_all = mastertable["label"].astype(int).to_numpy()
        X_tv, y_tv = X_all[trainval_mask], y_all[trainval_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        for model_name, model_cls in models.items():
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

            if USE_THRESHOLD_TUNING:
                tv_probs = best_model.predict_proba(X_tv)[:, 1]
                tv_labels = y_tv
                best_thresh, best_f1 = find_best_threshold(tv_probs, tv_labels)
            else:
                best_thresh = args.threshold

            proba = best_model.predict_proba(X_test)[:, 1]
            preds = (proba >= best_thresh).astype(int)
            auc = roc_auc_score(y_test, proba)
            precision = precision_score(y_test, preds, pos_label=1)
            recall = recall_score(y_test, preds, pos_label=1)
            f1_binary = f1_score(y_test, preds, average="binary", pos_label=1)

            df_roc = pd.DataFrame({"y_true": y_test, "y_score": proba})
            roc_path = os.path.join(
                OUT_DIR,
                f"roc_inputs_{model_name}_{args.modality}_fold{outer_fold}_thr{args.threshold}.csv"
            )
            pr_path = os.path.join(
                OUT_DIR,
                f"pr_inputs_{model_name}_{args.modality}_fold{outer_fold}_thr{args.threshold}.csv"
            )
            df_roc.to_csv(roc_path, index=False)
            df_roc.to_csv(pr_path, index=False)

            per_fold_metrics = {
                "Fold": outer_fold,
                "Model": model_name,
                "Precision_minority": precision,
                "Recall_minority": recall,
                "F1_minority": f1_binary,
                "AUC": auc
            }
            per_fold_df = pd.DataFrame([per_fold_metrics])
            per_fold_csv = os.path.join(OUT_DIR, f"{model_name}_fold_{outer_fold}_selected_metrics.csv")
            per_fold_df.to_csv(per_fold_csv, index=False)

            fold_key = f"{model_name}_fold_{outer_fold}"
            nested_cv_results[fold_key] = {
                "auc": auc,
                "precision_minority": precision,
                "recall_minority": recall,
                "f1_minority": f1_binary,
                "best_config": grid.best_params_
            }

    results = {}
    for fold_key, metrics in nested_cv_results.items():
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

    summary_path = os.path.join(OUT_DIR, "traditional_ml_nested_gridsearch_summary.csv")
    # Save summary metrics
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(OUT_DIR, "traditional_ml_nested_gridsearch_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary metrics to {summary_csv}")