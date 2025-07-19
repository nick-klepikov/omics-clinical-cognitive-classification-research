# SPDX-License-Identifier: MIT
# Adapted from Gómez de Lope et al., "Graph Representation Learning Strategies for Omics Data: A Case Study on Parkinson’s Disease", arXiv:2406.14442 (MIT License)

from src.classification_pipeline.utils.utils import *
from src.classification_pipeline.models.models import *
from sklearn.utils import class_weight
import yaml
import os
import argparse
import matplotlib
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
import time
start_time = time.time()
# ------------ Helper for inner CV (nested param search) ------------
def evaluate_config_on_fold(config, W, mastertable_file, outer_fold, param_set):
    """
    Given a config (with hyperparams set), affinity matrix W, mastertable file, outer CV fold,
    run inner StratifiedKFold CV on train+val set, train model on each inner fold, and return
    average validation AUC across inner folds.
    """
    # Load and filter mastertable as in main
    mastertable = pd.read_csv(mastertable_file, index_col="PATNO")
    # Filter features based on modality
    modality = config.get("modality", None)
    if modality is None:
        modality = "fused"  # fallback
    if modality == "geno":
        mastertable = mastertable[
            [col for col in mastertable.columns if not col.startswith("ENSG")]]
    elif modality == "rna":
        mastertable = mastertable[
            [col for col in mastertable.columns if
             col.startswith("ENSG") or col in ["label", "split", "age_at_visit", "SEX_M", "EDUCATION_YEARS"]]]
    # split masks
    split = mastertable["split"]
    test_mask = (split == "test").to_numpy()
    trainval_mask = (split != "test").to_numpy()
    trainval_idx = np.where(trainval_mask)[0]
    y = mastertable["label"].astype(int).to_numpy()
    features_df = mastertable.drop(columns=["label", "split"])
    features_name = features_df.columns
    X = features_df.to_numpy()
    X_indices = features_df.index
    folds = config["n_folds"]
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
    # For each inner fold, train model and record val AUC
    val_aucs = []
    for fold, (train_msk, val_msk) in enumerate(zip(raw_train, raw_val)):
        print(f"    Inner fold {fold} for outer fold {outer_fold}, config {param_set}")
        # set seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Build adjacency
        adj_df = pd.DataFrame(W, index=X_indices, columns=X_indices)
        adj = adj_df
        # create data_processing
        data = create_pyg_data(adj, pd.DataFrame(data=X, columns=features_name, index=X_indices), y, train_msk, val_msk, test_mask)
        if "GTC_uw" in config.get("model", ""):
            data.edge_attr = torch.ones((data.edge_index.shape[1], 1), device=data.edge_index.device)
        elif "GTC" in config.get("model", "") or "GINE" in config.get("model", ""):
            data.edge_attr = data.edge_attr.unsqueeze(-1)
        if "GPST" in config.get("model", ""):
            data.x, _ = pad_features(data.x, config["heads"], features_name)
        # Model
        model = generate_model(config.get("model", "GCNN"), config, data)
        model.apply(init_weights)
        model = model.to(check_cuda())
        train_labels = data.y[data.train_mask].cpu().numpy().astype(int)
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.array([0, 1]),
                                                          y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        criterion.to(check_cuda())
        # Ensure numeric types for hyperparameters
        lr = float(config["lr"])
        wd = float(config["weight_decay"])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["lrscheduler_factor"],
                                                   threshold=0.0001, patience=15,
                                                   verbose=True)
        n_epochs = int(config["n_epochs"])
        if "MLP" in model._get_name():
            data.edge_index = torch.empty((2, 0), dtype=torch.long).to(check_cuda())
            _, performance, best_epoch, _, _ = training_mlp_nowandb(check_cuda(), model, optimizer, scheduler, criterion, data, n_epochs, fold)
        else:
            _, performance, best_epoch, _, _ = training_nowandb(check_cuda(), model, optimizer, scheduler, criterion, data, n_epochs, fold)
        # Extract validation AUC at the best epoch (index 1 is validation)
        val_auc = float(performance['AUC'][best_epoch][1])
        val_aucs.append(val_auc)
        # Reset model params
        for name, module in model.named_children():
            if isinstance(module, torch.nn.ModuleList):
                for sub_module in module:
                    if hasattr(sub_module, 'reset_parameters'):
                        sub_module.reset_parameters()
            else:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
    return np.mean(val_aucs)

# ------------ Argument parsing ------------
parser = argparse.ArgumentParser(description="Train GCN on multi-omics data_processing")
parser.add_argument('--config',      type=str, required=True, help='Path to YAML config file')
parser.add_argument('--out_dir',     type=str, required=True, help='Directory for outputs')
parser.add_argument('--mastertable', type=str, required=True, help='Path to mastertable CSV')
parser.add_argument('--modality',   type=str, choices=["geno", "rna", "fused"], default="fused", help='Modality to train on')
parser.add_argument('--model', type=str, choices=["GCNN", "MLP2", "GAT"], default="GCNN")
parser.add_argument('--threshold', type=int, choices=[-2, -3, -4],  required=True, help='Threshold for binarizing MoCA change')
args = parser.parse_args()

# ------------ Load hyperparameters ------------
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    # Build list of hyperparameter configurations from grid
    param_list = list(ParameterGrid(config['param_grid']))

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

device = check_cuda()

# Set matplotlib to 'agg'
if matplotlib.get_backend() != 'agg':
    matplotlib.use('agg')

#----------------  Main function -----------------#
if __name__ == '__main__':
    for outer_fold in range(5):
        print(f"Starting outer fold {outer_fold}")
        # Update mastertable file and adjacency matrix path
        mastertable_file = f"{args.mastertable}_fold_{outer_fold}_thresh_{args.threshold}.csv"
        W = np.load(f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/fused_datasets/affinity_matrices/W_{args.modality}_fold_{outer_fold}_thresh_{args.threshold}.npy")
        # Sparsify W to top-k neighbors per row (excluding self)
        k = 10  # Fixed top-k value, can parameterize later
        W_sparse = np.zeros_like(W)
        for i in range(W.shape[0]):
            row = W[i].copy()
            row[i] = 0  # exclude self-loop
            topk_idx = np.argpartition(row, -k)[-k:]
            W_sparse[i, topk_idx] = W[i, topk_idx]
        W_sparse = np.maximum(W_sparse, W_sparse.T)  # Make symmetric
        W = W_sparse

        # Hyperparameter grid search (nested CV)
        best_auc = -np.inf
        best_params = None
        for config_idx, param_set in enumerate(param_list):
            print(f"  Outer fold {outer_fold} – evaluating config {config_idx + 1}/{len(param_list)}: {param_set}")
            # apply param_set to config
            for k_param, v_param in param_set.items():
                config[k_param] = v_param
            # compute inner-fold average validation AUC
            mean_val_auc = evaluate_config_on_fold(config, W, mastertable_file, outer_fold, param_set)
            if mean_val_auc > best_auc:
                best_auc, best_params = mean_val_auc, param_set.copy()
        # Re–apply best_params for final training on this outer fold
        for k_param, v_param in best_params.items():
            config[k_param] = v_param
        # Save this fold’s best hyperparameters
        params_out = os.path.join(OUT_DIR, f"outer_{outer_fold}_best_params.yaml")
        with open(params_out, 'w') as pf:
            yaml.dump(best_params, pf)

        # ---- Final train/val/test on outer fold with best hyperparameters ----
        # Load and filter mastertable
        mastertable = pd.read_csv(mastertable_file, index_col="PATNO")
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
        y = mastertable["label"].astype(int)
        features_df = mastertable.drop(columns=["label", "split"])
        feat_names = features_df.columns.tolist()
        labels_dict = y.to_dict()
        features_name = features_df.columns
        y = y.to_numpy()
        X = features_df.to_numpy()
        X_indices = features_df.index
        pos = get_pos_similarity(features_df)
        folds = config["n_folds"]
        fold_v_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [], 'N_epoch': [],
                              'Loss': [], 'N_features': [], 'homophily_index': []}
        fold_test_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [],
                                 'N_epoch': []}
        sub_y = y[trainval_mask]
        trainval_indices = np.where(trainval_mask)[0]
        # Use a single 10% stratified hold-out for final validation
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        # Single 90% train / 10% val split
        train_idx, val_idx = next(sss.split(np.zeros(len(sub_y)), sub_y))
        mask_train = np.zeros(len(y), dtype=bool)
        mask_val   = np.zeros(len(y), dtype=bool)
        mask_train[trainval_indices[train_idx]] = True
        mask_val[  trainval_indices[val_idx]]   = True
        # Assign single masks
        train_msk = mask_train
        val_msk   = mask_val
        # We only run this once:
        # Only one fold here, so we skip enumeration
        fold = 0
        X_train, X_val = X[train_msk], X[val_msk]
        y_train, y_val = y[train_msk], y[val_msk]
        adj_df = pd.DataFrame(W, index=X_indices, columns=X_indices)
        adj = adj_df
        data = create_pyg_data(adj, pd.DataFrame(data=X, columns=features_name, index=X_indices), y, train_msk, val_msk, test_mask)

        if "GTC_uw" in args.model:
            data.edge_attr = torch.ones((data.edge_index.shape[1], 1), device=data.edge_index.device)
        elif "GTC" in args.model or "GINE" in args.model:
            data.edge_attr = data.edge_attr.unsqueeze(-1)
        if "GPST" in args.model:
            data.x, feat_names = pad_features(data.x, config["heads"], feat_names)
        # homophily_index = homophily(data_processing.edge_index, data_processing.y, method='edge')
        # print(f'Homophily index: {homophily_index}')
        model = generate_model(args.model, config, data)
        model.apply(init_weights)
        model = model.to(device)
        train_labels = data.y[data.train_mask].cpu().numpy().astype(int)
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.array([0, 1]),
                                                          y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        criterion.to(device)
        # Ensure numeric types for hyperparameters
        lr_val = float(config["lr"])
        wd_val = float(config["weight_decay"])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=wd_val)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["lrscheduler_factor"],
                                                   threshold=0.0001, patience=15,
                                                   verbose=True)
        n_epochs = int(config["n_epochs"])
        if "MLP" in model._get_name():
            data.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
            losses, performance, best_epoch, best_loss, best_model = training_mlp_nowandb(device, model, optimizer, scheduler, criterion, data, n_epochs, fold)
        else:
            losses, performance, best_epoch, best_loss, best_model = training_nowandb(device, model, optimizer,
                                                                        scheduler, criterion, data,
                                                                        n_epochs, fold)
        # ---- record metrics for this fold at its best epoch ----
        # Validation metrics at best epoch
        fold_v_performance["Fold"].append(fold)
        fold_v_performance["N_epoch"].append(best_epoch)
        for m in ["AUC", "Accuracy", "Recall", "Specificity", "F1"]:
            # performance[m] is a list of [train, val, test] tuples
            fold_v_performance[m].append(performance[m][best_epoch][1])
        # Test metrics at best epoch
        fold_test_performance["Fold"].append(fold)
        fold_test_performance["N_epoch"].append(best_epoch)
        for m in ["AUC", "Accuracy", "Recall", "Specificity", "F1"]:
            fold_test_performance[m].append(performance[m][best_epoch][2])
        # reset parameters
        for name, module in model.named_children():
            if isinstance(module, torch.nn.ModuleList):
                for sub_module in module:
                    if hasattr(sub_module, 'reset_parameters'):
                        sub_module.reset_parameters()
            else:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

        selected_metrics = {
            'Fold': outer_fold,
            'Epoch': fold_test_performance["N_epoch"][0],
            'AUC':  fold_test_performance["AUC"][0],
            'Accuracy':  fold_test_performance["Accuracy"][0],
            'Recall': fold_test_performance["Recall"][0],
            'Specificity': fold_test_performance["Specificity"][0],
            'F1': fold_test_performance["F1"][0]
        }
        pd.DataFrame([selected_metrics]).to_csv(os.path.join(OUT_DIR, f"fold_{outer_fold}_selected_metrics.csv"), index=False)


    # --- Performance-based final hyperparameter selection ---
    # Read per-fold best params and test AUCs
    records = []
    for fold in range(5):
        # load params
        params = yaml.safe_load(open(os.path.join(OUT_DIR, f"outer_{fold}_best_params.yaml")))
        # load test metrics
        df = pd.read_csv(os.path.join(OUT_DIR, f"fold_{fold}_selected_metrics.csv"))
        auc = df.loc[0, 'AUC']
        record = {**params, 'TestAUC': auc, 'fold': fold}
        records.append(record)
    df_all = pd.DataFrame(records)
    # group by hyperparams to compute mean TestAUC and frequency
    param_cols = [c for c in df_all.columns if c not in ['TestAUC', 'fold']]
    summary = (
        df_all
        .groupby(param_cols)
        .agg(count=('fold', 'size'),
             mean_auc=('TestAUC', 'mean'))
        .reset_index()
        .sort_values(['mean_auc', 'count'], ascending=False)
    )
    # pick the best config
    best_row = summary.iloc[0]
    # Extract and cast each param to native Python
    final_params = {}
    for c in param_cols:
        val = best_row[c]
        # unwrap numpy scalar or pandas type
        if hasattr(val, 'item'):
            val = val.item()
        final_params[c] = val
    # include mean AUC
    final_params['mean_test_AUC'] = float(best_row['mean_auc'])
    # save final params with safe dump
    final_out = os.path.join(OUT_DIR, "config_ablation_dos_gnn_threshold_-3.yaml")
    with open(final_out, 'w') as f:
        yaml.safe_dump(final_params, f, default_flow_style=False)

    # Print only the total elapsed time at the end
    elapsed = time.time() - start_time
    print(f"Total pipeline runtime: {elapsed:.1f} seconds")
