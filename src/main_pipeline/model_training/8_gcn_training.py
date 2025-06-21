# SPDX-License-Identifier: MIT
# Adapted from Gómez de Lope et al., "Graph Representation Learning Strategies for Omics Data: A Case Study on Parkinson’s Disease", arXiv:2406.14442 (MIT License)

import numpy as np
import pandas as pd
import random
import os
from matplotlib import pyplot as plt
from functions import *
from functions import stratified_continuous_folds
from model import *
from sklearn.utils import class_weight
import networkx as nx
import wandb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pathlib import Path
import shap
import matplotlib


#wandb.login()
#sweep_id = wandb.sweep(sweep=sweep_config, project='my-cv-test5')
# os.environ["WANDB_API_KEY"] = "" set up environment variable??

os.environ["WANDB_MODE"] = "disabled"

# Hard-coded hyperparameters in lieu of external YAML
DEFAULT_CONFIG = {
    "C_lasso":             0.5,
    "n_folds":             5,
    "S_threshold":         0.32,
    "n_epochs":            250,
    "lr":                  1e-3,
    "weight_decay":        1e-4,
    "cl1_hidden_units":    64,
    "cl2_hidden_units":    16,
    "ll_out_units":        1,
    "dropout":             0.3,
    "lrscheduler_factor":  0.5,
    "model_name":          "GCNN",
    "heads":               3,
    "K_cheby":             2,
}

# Initialize W&B locally using DEFAULT_CONFIG
sweep_run = wandb.init(config=DEFAULT_CONFIG,
                       project="gcn",
                       name="local-run",
                       reinit=True)
myconfig = sweep_run.config

# Set matplotlib to 'agg'
if matplotlib.get_backend() != 'agg':
    print(f"Switching Matplotlib backend from '{matplotlib.get_backend()}' to 'agg'")
    matplotlib.use('agg')


device = check_cuda()

#----------------  Main function -----------------#
if __name__ == '__main__':
    # set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set other params
    device = check_cuda()
    # I/O
    OUT_DIR = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/final_res/" + myconfig.model_name + "/"
    # Clean output directory if it exists, otherwise create it
    if os.path.exists(OUT_DIR):
        for entry in os.scandir(OUT_DIR):
            if entry.is_file():
                os.remove(entry.path)
    else:
        os.makedirs(OUT_DIR)
    mastertable_file = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/fused_datasets/final_mastertable.csv"
    # Load your pre-selected 200-feature master table
    mastertable = pd.read_csv(mastertable_file, index_col=0)
    print("Dimensions of mastertable:", mastertable.shape)  # expected (n_samples, 201)
    # extract labels and features
    y = mastertable["moca_change"].astype(float)
    features_df = mastertable.drop(columns=["moca_change"])
    # prepare label dict for plotting
    labels_dict = y.to_dict()
    # feature names, positions, and numpy arrays
    features_name = features_df.columns
    pos = get_pos_similarity(features_df)
    y = y.to_numpy()
    X = features_df.to_numpy()
    X_indices = features_df.index
    # cross-validation
    folds = myconfig.n_folds
    # Use regression metrics for performance tracking
    fold_v_performance = {'Fold': [], 'MSE': [], 'MAE': [], 'R2': [], 'N_epoch': [],
                          'Loss': [], 'N_features': [],
                          'Baseline_MSE': [], 'Baseline_MAE': [], 'Baseline_R2': []}
    features_track = {'Fold': [], 'N_selected_features': [], 'Selected_Features': [], 'Relevant_Features': []}
    # get back the two lists
    train_masks, val_masks = stratified_continuous_folds(y, folds)

    # now iterate pairwise
    for fold, (train_msk, val_msk) in enumerate(zip(train_masks, val_masks)):
        # define data splits
        X_train, X_val = X[train_msk], X[val_msk]
        y_train, y_val = y[train_msk], y[val_msk]

        # --- Baseline: constant-mean predictor ---
        # always predict the mean of the training labels
        y_train_mean = y_train.mean()
        # create constant predictions for validation set
        y_pred_baseline = np.full_like(y_val, fill_value=y_train_mean, dtype=float)
        # compute baseline metrics
        mse_base = mean_squared_error(y_val, y_pred_baseline)
        mae_base = mean_absolute_error(y_val, y_pred_baseline)
        r2_base  = r2_score(y_val, y_pred_baseline)
        print(f'Fold {fold} constant-mean baseline — MSE: {mse_base:.4f}, MAE: {mae_base:.4f}, R2: {r2_base:.4f}')
        # log baseline metrics
        wandb.log({
            f'baseline/MSE-{fold}': mse_base,
            f'baseline/MAE-{fold}': mae_base,
            f'baseline/R2-{fold}':  r2_base,
        })
        # track baseline in fold_v_performance
        fold_v_performance['Baseline_MSE'].append(mse_base)
        fold_v_performance['Baseline_MAE'].append(mae_base)
        fold_v_performance['Baseline_R2'].append(r2_base)
        # Load fused adjacency matrix from SNF output
        # Adjust path to point to your W_fused.npy
        fused_path = "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/fused_datasets/W_fused.npy"
        W = np.load(fused_path)  # shape (n_samples, n_samples)

        # Convert to DataFrame with PATNO index and columns
        adj_df = pd.DataFrame(W, index=X_indices, columns=X_indices)

        # Use this as adjacency
        adj = adj_df

        # If you still need a NetworkX graph:
        G = nx.from_pandas_adjacency(adj_df)

        # create graph data object
        data = create_pyg_data(adj, pd.DataFrame(data=X, columns=features_name, index=X_indices), y, train_msk, val_msk)


        # model
        print(myconfig.model_name)
        model = generate_model(myconfig.model_name, myconfig, data)
        model.apply(init_weights)
        model = model.to(device)
        # regression loss
        criterion = nn.MSELoss()
        criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=myconfig.lr, weight_decay=myconfig.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=myconfig.lrscheduler_factor,
                                                   threshold=0.0001, patience=15,
                                                   verbose=True)
        n_epochs = myconfig.n_epochs
        # training for graph methods
        losses, performance, best_epoch, best_loss, best_model = training(device, model, optimizer, scheduler, criterion, data, n_epochs, fold, wandb)
        # feature importance
        feature_importance = feature_importance_gnnexplainer(model, data, names_list=features_name.tolist(), save_fig=True, name_file=f'{sweep_run.name}-{fold}_feature_importance',path=OUT_DIR)
        feature_importance = feature_importance.index.tolist()
        fold_v_performance, features_track = update_overall_metrics(fold, best_epoch, features_name.tolist(), feature_importance, performance, losses, fold_v_performance, features_track)
        # log performance and loss in wandb
        eval_info = {
            f'best_val_MSE-{fold}': losses[best_epoch][1],
            f'best_val_R2-{fold}': performance["R2"][best_epoch][1]
        }
        wandb.log(eval_info)
        # reset parameters
        print('*resetting model parameters*')
        for name, module in model.named_children():
            if isinstance(module, torch.nn.ModuleList):
                for sub_module in module:
                    if hasattr(sub_module, 'reset_parameters'):
                        sub_module.reset_parameters()
            else:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
    cv_metrics_to_wandb(fold_v_performance)
    print("sweep", sweep_run.name, pd.DataFrame.from_dict(fold_v_performance))
    print("sweep", sweep_run.name, pd.DataFrame.from_dict(features_track))
    # exports & plots performance & losses
    pd.DataFrame.from_dict(features_track).to_csv(OUT_DIR + sweep_run.name + "_features_track.csv", index=False)
    pd.DataFrame.from_dict(fold_v_performance).to_csv(OUT_DIR + sweep_run.name + "_val_performance.csv", index=False)
