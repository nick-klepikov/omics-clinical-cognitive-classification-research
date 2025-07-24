from torch.optim import lr_scheduler

from src.classification_pipeline.utils.utils import *
from src.classification_pipeline.models.models import *
import yaml
from torch_geometric.utils import homophily
import argparse
import matplotlib
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import pandas as pd
import glob
import time
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score, f1_score,
    average_precision_score
)


def find_best_threshold(probs, labels, metric=f1_score):
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_thresh = 0.5
    best_score = -1
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = metric(labels, preds)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score


parser = argparse.ArgumentParser(description="Train GCN on multi-omics data_processing")
parser.add_argument('--config',      type=str, required=True, help='Path to YAML config file')
parser.add_argument('--out_dir',     type=str, required=True, help='Directory for outputs')
parser.add_argument('--mastertable', type=str, required=True, help='Path to mastertable CSV')
parser.add_argument('--modality',   type=str, choices=["geno", "rna", "fused"], default="fused", help='Modality to train on')
parser.add_argument('--model', type=str, choices=["GCNN", "MLP2", "GAT", "DOS_GNN"], default="GCNN")
parser.add_argument('--threshold', type=int, choices=[-2, -3, -4, -5],  required=True, help='Threshold for binarizing MoCA change')
args = parser.parse_args()

USE_SPARSE_GRAPH = True
USE_WEIGHTED_LOSS = False
USE_THRESHOLD_TUNING = False

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

model_name = args.model
modality_name = args.modality
threshold_value = args.threshold

device = check_cuda()

if matplotlib.get_backend() != 'agg':
    matplotlib.use('agg')

if __name__ == '__main__':
    total_start = time.time()
    outer_fold_medians = {
        'outer_fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [], 'AveragePrecision': []
    }

    for outer_fold in range(5):
        fold_start = time.time()
        # set seeds
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load SNP, RNA, and clinical feature matrices
        mastertable_file = f"{args.mastertable}_fold_{outer_fold}_thresh_{args.threshold}.csv"
        W = np.load(f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/fused_datasets/affinity_matrices/W_{args.modality}_fold_{outer_fold}_thresh_{args.threshold}.npy")

        # Construct sample similarity graph
        # Sparsify W if enabled
        k = 50 if USE_SPARSE_GRAPH else 0
        if k > 0:
            W_sparse = np.zeros_like(W)
            for i in range(W.shape[0]):
                row = W[i].copy()
                row[i] = 0
                topk_idx = np.argpartition(row, -k)[-k:]
                W_sparse[i, topk_idx] = W[i, topk_idx]
            W_sparse = np.maximum(W_sparse, W_sparse.T)
            W = W_sparse

        mastertable = pd.read_csv(mastertable_file, index_col="PATNO")

        # Filter features by modality
        if args.modality == "geno":
            mastertable = mastertable[
                [col for col in mastertable.columns if not col.startswith("ENSG")]]
        elif args.modality == "rna":
            mastertable = mastertable[
                [col for col in mastertable.columns if
                 col.startswith("ENSG") or col in ["label", "split", "age_at_visit", "SEX_M", "EDUCATION_YEARS"]]]

        split = mastertable["split"]
        test_mask = (split == "test").to_numpy()
        trainval_mask = (split != "test").to_numpy()
        trainval_idx = np.where(trainval_mask)[0]

        print(f"Outer fold {outer_fold}")  # Keep outer fold number print
        y = mastertable["label"].astype(int)

        split_series = split  # reuse split variable
        mask_tv = split_series != "test"
        y_tv = y[mask_tv]
        mask_test = split_series == "test"
        y_test = y[mask_test]
        if y_test.size == 0:
            print("No samples in test split")
        features_df = mastertable.drop(columns=["label", "split"])
        feat_names = features_df.columns.tolist()
        print(f"Using {len(feat_names)} features for training and evaluation.")

        features_name = features_df.columns
        y = y.to_numpy()
        X = features_df.to_numpy()
        X_indices = features_df.index

        folds = config["n_folds"]
        fold_v_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [], 'N_epoch': [],
                              'Loss': [], 'N_features': [], 'homophily_index': []}
        fold_test_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [],
                                 'N_epoch': []}
        features_track = {'Fold': [], 'N_selected_features': [], 'Relevant_Features': [], 'Raw_Feature_Importances': []}

        sub_y = y[trainval_mask]
        trainval_indices = np.where(trainval_mask)[0]
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        raw_train = []
        raw_val = []
        for train_idx, val_idx in skf.split(np.zeros(len(sub_y)), sub_y):
            mask_train = np.zeros(len(y), dtype=bool)
            mask_val = np.zeros(len(y), dtype=bool)
            mask_train[trainval_indices[train_idx]] = True
            mask_val[trainval_indices[val_idx]] = True
            raw_train.append(mask_train)
            raw_val.append(mask_val)

        roc_probs_list = []
        roc_labels_list = []
        inner_thresholds = []
        # Train selected model (GNN or traditional ML)
        for fold, (train_msk, val_msk) in enumerate(zip(raw_train, raw_val)):
            X_train, X_val = X[train_msk], X[val_msk]
            y_train, y_val = y[train_msk], y[val_msk]
            adj_df = pd.DataFrame(W, index=X_indices, columns=X_indices)
            adj = adj_df
            data = create_pyg_data(adj, pd.DataFrame(data=X, columns=features_name, index=X_indices), y, train_msk, val_msk,
                                   test_mask)
            # Homophily calculation (short comment)
            H_raw = homophily(data.edge_index, data.y, method='edge')
            y_all = data.y.cpu().numpy()
            classes, counts = np.unique(y_all, return_counts=True)
            props = counts.astype(float) / counts.sum()
            H_baseline = np.sum(props ** 2)
            H_norm = (H_raw - H_baseline) / (1.0 - H_baseline) if (1.0 - H_baseline) > 0 else 0.0
            homophily_index = H_norm
            # Minority-class homophily
            edges = data.edge_index.cpu().numpy().T
            y_all = data.y.cpu().numpy()
            classes, counts = np.unique(y_all, return_counts=True)
            minor_label = classes[np.argmin(counts)]
            mask_touch_min = (y_all[edges[:,0]] == minor_label) | (y_all[edges[:,1]] == minor_label)
            edges_min = edges[mask_touch_min]
            if edges_min.shape[0] > 0:
                same_min = (y_all[edges_min[:,0]] == y_all[edges_min[:,1]]).sum()
                r_min = same_min / edges_min.shape[0]
            else:
                r_min = 0.0

            model = generate_model(args.model, config, data)
            model.apply(init_weights)
            model = model.to(device)

            if USE_WEIGHTED_LOSS:
                from sklearn.utils.class_weight import compute_class_weight
                y_train = data.y[data.train_mask].cpu().numpy()
                unique_classes = np.unique(y_train)
                class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
                class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
            else:
                criterion = nn.CrossEntropyLoss(reduction="mean")
            criterion.to(device)

            lr = float(config["lr"])
            wd = float(config["weight_decay"])
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["lrscheduler_factor"],
                                                       threshold=0.0001, patience=15,
                                                       verbose=True)
            n_epochs = int(config["n_epochs"])
            if "MLP" in model._get_name():
                data.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
                losses, performance, best_epoch, best_loss, best_model = training_mlp_nowandb(device, model, optimizer, scheduler, criterion, data, n_epochs, fold)
            elif "DOS" in model._get_name().upper():
                (losses, performance, best_epoch, best_loss, best_model,
                 head_losses, head_perf, head_best_epoch, head_best_loss, best_head) = training_nowandb_dos_gnn(
                    device, model, optimizer, scheduler, criterion, data, n_epochs, fold)
            else:
                losses, performance, best_epoch, best_loss, best_model = training_nowandb(device, model, optimizer,
                                                                                scheduler, criterion, data,
                                                                                n_epochs, fold)
            feat_imp_ser = feature_importance_gnnexplainer(
                model,
                data,
                names_list=feat_names,
                save_fig=True,
                name_file=f'fold_{outer_fold}_{fold}_feature_importance',
                path=OUT_DIR
            )
            raw_imp = feat_imp_ser['Importance_score'].to_dict()
            feat_labels = feat_imp_ser.index.tolist()

            fold_v_performance, fold_test_performance, features_track = update_overall_metrics(
                fold, best_epoch, homophily_index, feat_names, feat_labels, raw_imp, performance, losses,
                fold_v_performance, fold_test_performance, features_track
            )

            # Evaluate model and collect metrics
            model = best_model
            model.eval()
            # Tune threshold on validation set only
            if USE_THRESHOLD_TUNING:
                # get probabilities for validation mask
                if "MLP" in model._get_name():
                    val_logits = model(data.x.to(device).float(), data.edge_index.to(device))
                else:
                    val_logits = model(
                        data.x.to(device).float(),
                        data.edge_index.to(device),
                        data.edge_attr.to(device).float()
                    )
                val_probs = torch.softmax(val_logits[data.val_mask], dim=1)[:, 1].detach().cpu().numpy()
                val_labels = data.y[data.val_mask].cpu().numpy()
                best_thresh, best_f1 = find_best_threshold(val_probs, val_labels, metric=f1_score)
                inner_thresholds.append(best_thresh)
                print(f"[Threshold Tuning] Best threshold (validation): {best_thresh:.2f}, F1: {best_f1:.3f}")
            else:
                inner_thresholds.append(0.5)
            with torch.no_grad():
                if "MLP" in model._get_name():
                    logits = model(data.x.to(device).float(), data.edge_index.to(device))
                else:
                    logits = model(
                        data.x.to(device).float(),
                        data.edge_index.to(device),
                        data.edge_attr.to(device).float()
                    )
                probs = torch.softmax(logits[data.test_mask], dim=1)[:, 1].detach().cpu().numpy()
                labels = data.y[data.test_mask].cpu().numpy()
                patnos = np.array(data.PATNO)[data.test_mask.cpu().numpy()] if hasattr(data, "PATNO") else np.arange(len(labels))

            roc_probs_list.append(probs)
            roc_labels_list.append(labels)

            for name, module in model.named_children():
                if isinstance(module, torch.nn.ModuleList):
                    for sub_module in module:
                        if hasattr(sub_module, 'reset_parameters'):
                            sub_module.reset_parameters()
                else:
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

        auc_list = fold_test_performance['AUC']
        median_auc = np.median(auc_list)
        idx = int(np.argmin([abs(a - median_auc) for a in auc_list]))
        print(f"Outer fold {outer_fold}: median AUC = {median_auc:.4f}, selected inner fold = {idx}")
        best_epoch = fold_v_performance['N_epoch'][idx]
        median_labels = roc_labels_list[idx]
        median_probs  = roc_probs_list[idx]
        pd.DataFrame({"y_true": median_labels, "y_score": median_probs}).to_csv(
            os.path.join(
                OUT_DIR,
                f"roc_inputs_{args.model}_{args.modality}_fold{outer_fold}_thr{args.threshold}.csv"
            ),
            index=False
        )
        pd.DataFrame({"y_true": median_labels, "y_score": median_probs}).to_csv(
            os.path.join(
                OUT_DIR,
                f"pr_inputs_{args.model}_{args.modality}_fold{outer_fold}_thr{args.threshold}.csv"
            ),
            index=False
        )

        # Use a single threshold from validation tuning, not per-test tuning
        if USE_THRESHOLD_TUNING:
            median_thresh = np.median(inner_thresholds)
        else:
            median_thresh = 0.5
        test_probs = roc_probs_list[idx]
        test_labels = roc_labels_list[idx]
        test_preds = (test_probs >= median_thresh).astype(int)
        acc = accuracy_score(test_labels, test_preds)
        rec = recall_score(test_labels, test_preds)
        prec = precision_score(test_labels, test_preds)
        auc = roc_auc_score(test_labels, test_probs)
        f1 = f1_score(test_labels, test_preds)
        ap = average_precision_score(test_labels, test_probs)
        print(f"Test F1 at threshold {median_thresh:.2f}: {f1:.3f}")

        selected_metrics = {
            'Fold': outer_fold,
            'Selected_Fold': idx,
            'Epoch': best_epoch,
            'AUC': auc,
            'Accuracy': acc,
            'Recall': rec,
            'Specificity': fold_test_performance['Specificity'][idx],
            'F1': f1,
            'AveragePrecision': ap
        }
        pd.DataFrame([selected_metrics]).to_csv(os.path.join(OUT_DIR, f"fold_{outer_fold}_selected_metrics.csv"), index=False)
        selected_raw_imp = features_track['Raw_Feature_Importances'][idx]
        df_imp = pd.DataFrame.from_dict(
            selected_raw_imp, orient='index', columns=['Importance_score']
        ).reset_index().rename(columns={'index':'Feature'})
        df_imp.to_csv(os.path.join(OUT_DIR, f"fold_{outer_fold}_selected_feature_importances.csv"), index=False)
        outer_fold_medians['outer_fold'].append(outer_fold)
        outer_fold_medians['AUC'].append(selected_metrics['AUC'])
        outer_fold_medians['Accuracy'].append(selected_metrics['Accuracy'])
        outer_fold_medians['Recall'].append(selected_metrics['Recall'])
        outer_fold_medians['Specificity'].append(selected_metrics['Specificity'])
        outer_fold_medians['F1'].append(selected_metrics['F1'])
        outer_fold_medians['AveragePrecision'].append(selected_metrics['AveragePrecision'])


    # Final summary and AUC consistency check
    summary = {}
    for m in ['AUC', 'Accuracy', 'Recall', 'Specificity', 'F1', 'AveragePrecision']:
        summary[f"{m}_mean"] = np.mean(outer_fold_medians[m])
        summary[f"{m}_std"] = np.std(outer_fold_medians[m])
    sparsity_suffix = "_sparse" if USE_SPARSE_GRAPH else "_full"
    summary_filename = f"{model_name}_{modality_name}_thr{threshold_value}{sparsity_suffix}_ablation_final_summary.csv"
    pd.DataFrame([summary]).to_csv(
        os.path.join(OUT_DIR, summary_filename),
        index=False
    )
    pattern = os.path.join(
        OUT_DIR,
        f"roc_inputs_{model_name}_{modality_name}_fold*_thr{threshold_value}.csv"
    )
    roc_files = sorted(glob.glob(pattern))
    plot_aucs = []
    for rf in roc_files:
        df_roc = pd.read_csv(rf)
        plot_aucs.append(roc_auc_score(df_roc['y_true'], df_roc['y_score']))
    mean_plot_auc = np.mean(plot_aucs)
    std_plot_auc  = np.std(plot_aucs)
    print(f"Summary AUC    = {summary['AUC_mean']:.3f} ± {summary['AUC_std']:.3f}")
    print(f"Plot-based AUC = {mean_plot_auc:.3f} ± {std_plot_auc:.3f}")

    # Aggregate feature importances across folds
    try:
        paths = glob.glob(os.path.join(OUT_DIR, "fold_*_selected_feature_importances.csv"))
        dfs = []
        for p in paths:
            df = pd.read_csv(p)
            fold = int(os.path.basename(p).split("_")[1])
            df["OuterFold"] = fold
            dfs.append(df)
        all_imp = pd.concat(dfs, ignore_index=True)
        counts = all_imp.groupby("Feature")["OuterFold"].nunique()
        stable = counts[counts >= 2].index
        filtered_imp = all_imp[all_imp["Feature"].isin(stable)]
        summary_imp = (
            filtered_imp
            .groupby("Feature")["Importance_score"]
            .agg(FoldCount="nunique", MeanImportance="mean", StdImportance="std")
            .sort_values("MeanImportance", ascending=False)
            .reset_index()
        )
        imp_filename = f"{model_name}_{modality_name}_thr{threshold_value}{sparsity_suffix}_final_feature_ranking.csv"
        summary_imp.to_csv(
            os.path.join(OUT_DIR, imp_filename),
            index=False
        )
        print(f"Saved final feature ranking to {imp_filename}")
    except ValueError:
        print("Warning: No feature importance files to aggregate.")

    # Perform enrichment analysis on stable features

    # Generate and save ROC and PR plots
    patterns = [
        "fold_*_selected_feature_importances.csv",
    ]
    for pat in patterns:
        for f in glob.glob(os.path.join(OUT_DIR, pat)):
            os.remove(f)
