# SPDX-License-Identifier: MIT
# Adapted from Gómez de Lope et al., "Graph Representation Learning Strategies for Omics Data: A Case Study on Parkinson’s Disease", arXiv:2406.14442 (MIT License)

from src.main_pipeline.ablation.functions import *
from src.main_pipeline.ablation.models import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import yaml


# ------------ Argument parsing ------------
parser = argparse.ArgumentParser(description="Train GCN on multi-omics data")
parser.add_argument('--config',      type=str, required=True, help='Path to YAML config file')
parser.add_argument('--out_dir',     type=str, required=True, help='Directory for outputs')
parser.add_argument('--mastertable', type=str, required=True, help='Path to mastertable CSV')
parser.add_argument('--fused_adj',   type=str, required=True, help='Path to fused adjacency .npy file')
args = parser.parse_args()

# ------------ Load hyperparameters ------------
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

mastertable_file = args.mastertable
W = np.load(args.fused_adj)

# Sparsify W to top-k neighbors per row (excluding self)
k = 10  # Fixed top-k value, can parameterize later
W_sparse = np.zeros_like(W)
for i in range(W.shape[0]):
    row = W[i].copy()
    row[i] = -np.inf  # exclude self-loop
    topk_idx = np.argpartition(row, -k)[-k:]
    W_sparse[i, topk_idx] = W[i, topk_idx]
W_sparse = np.maximum(W_sparse, W_sparse.T)  # Make symmetric
W = W_sparse

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

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


    # Clean output directory if it exists, otherwise create it
    if os.path.exists(OUT_DIR):
        for entry in os.scandir(OUT_DIR):
            if entry.is_file():
                os.remove(entry.path)
    else:
        os.makedirs(OUT_DIR)

    mastertable = pd.read_csv(mastertable_file, index_col="PATNO")

    split = mastertable["split"]
    test_mask = (split == "test").to_numpy()  # Boolean mask over ALL samples
    trainval_mask = (split != "test").to_numpy()
    trainval_idx = np.where(trainval_mask)[0]  # integer indices of train+val rows

    print("Dimensions of mastertable:", mastertable.shape)  # expected (n_samples, 201)
    # extract labels and features
    y = mastertable["moca_change"].astype(float)
    features_df = mastertable.drop(columns=["moca_change", "split"])
    # prepare label dict for plotting
    labels_dict = y.to_dict()
    # feature names and numpy arrays
    features_name = features_df.columns
    y = y.to_numpy()
    X = features_df.to_numpy()
    X_indices = features_df.index


    # cross-validation
    folds = config["n_folds"]
    # Use regression metrics for performance tracking
    fold_v_performance = {'Fold': [], 'MSE': [], 'MAE': [], 'R2': [], 'N_epoch': [], 'N_features': [],
                          'Baseline_MSE': [], 'Baseline_MAE': [], 'Baseline_R2': []}
    fold_test_performance = {'Fold': [], 'MSE': [], 'MAE': [], 'R2': [], 'N_epoch': [],
                          'N_features': [],
                          'Baseline_MSE': [], 'Baseline_MAE': [], 'Baseline_R2': []}
    features_track = {'Fold': [], 'N_selected_features': [], 'Selected_Features': [], 'Relevant_Features': []}
    # get back the two lists

    # 1) extract only train+val labels
    sub_y = y[trainval_mask]

    # 2) do stratified CV on that subset
    raw_train, raw_val = stratified_continuous_folds(sub_y, folds)

    # Now “scatter” each relative mask back into a full-length boolean array
    train_fold_masks = []
    val_fold_masks = []

    for rel_train_idx, rel_val_idx in zip(raw_train, raw_val):
        # Start with all-False masks over the entire dataset
        current_train_mask = np.zeros_like(test_mask, dtype=bool)
        current_validation_mask = np.zeros_like(test_mask, dtype=bool)

        # Turn on only the train+val positions for this fold
        current_train_mask[trainval_idx[rel_train_idx]] = True
        current_validation_mask[trainval_idx[rel_val_idx]] = True

        train_fold_masks.append(current_train_mask)
        val_fold_masks.append(current_validation_mask)

    # now iterate pairwise
    for fold, (train_msk, val_msk) in enumerate(zip(train_fold_masks, val_fold_masks)):
        # define data splits
        X_train, X_val = X[train_msk], X[val_msk]
        y_train, y_val = y[train_msk], y[val_msk]

        # --- Baseline: constant-mean predictor ---
        # always predict the mean of the training labels
        y_train_mean = y_train.mean()
        # create constant predictions for validation set
        y_pred_baseline = np.full_like(y_val, fill_value=y_train_mean, dtype=float)
        # compute baseline metrics for validation
        mse_base = mean_squared_error(y_val, y_pred_baseline)
        mae_base = mean_absolute_error(y_val, y_pred_baseline)
        r2_base  = r2_score(y_val, y_pred_baseline)
        print(f'Fold {fold} constant-mean baseline — MSE: {mse_base:.4f}, MAE: {mae_base:.4f}, R2: {r2_base:.4f}')

        # track baseline in fold_v_performance
        fold_v_performance['Baseline_MSE'].append(mse_base)
        fold_v_performance['Baseline_MAE'].append(mae_base)
        fold_v_performance['Baseline_R2'].append(r2_base)

        # --- Baseline: constant-mean predictor on TEST set ---
        y_test = y[test_mask]
        y_pred_baseline_test = np.full_like(y_test, fill_value=y_train_mean, dtype=float)
        mse_base_test = mean_squared_error(y_test, y_pred_baseline_test)
        mae_base_test = mean_absolute_error(y_test, y_pred_baseline_test)
        r2_base_test = r2_score(y_test, y_pred_baseline_test)
        print(f'Fold {fold} constant-mean baseline on TEST — MSE: {mse_base_test:.4f}, MAE: {mae_base_test:.4f}, R2: {r2_base_test:.4f}')
        fold_test_performance['Baseline_MSE'].append(mse_base_test)
        fold_test_performance['Baseline_MAE'].append(mae_base_test)
        fold_test_performance['Baseline_R2'].append(r2_base_test)

        # Convert to DataFrame with PATNO index and columns
        adj_df = pd.DataFrame(W, index=X_indices, columns=X_indices)

        # Use this as adjacency
        adj = adj_df

        # create graph data object
        data = create_pyg_data(adj, pd.DataFrame(data=X, columns=features_name, index=X_indices), y, train_msk, val_msk, test_mask)


        # model
        print(config["model_name"])
        model = generate_model(config["model_name"], config, data)
        model.apply(init_weights)
        model = model.to(device)
        # regression loss
        criterion = nn.MSELoss()
        criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["lrscheduler_factor"],
                                                   threshold=0.0001, patience=15,
                                                   verbose=True)
        n_epochs = config["n_epochs"]
        # training for graph methods
        losses, performance, best_epoch, best_loss, best_model = training_nowandb(device, model, optimizer, scheduler, criterion, data, n_epochs, fold)
        # feature importance
        feature_importance = feature_importance_gnnexplainer(model, data, names_list=features_name.tolist(), save_fig=True, name_file=f'{fold}_feature_importance',path=OUT_DIR)
        feature_importance = feature_importance.index.tolist()
        fold_v_performance, fold_test_performance, features_track = update_overall_metrics(fold, best_epoch, features_name.tolist(), feature_importance, performance, losses, fold_v_performance, fold_test_performance, features_track)

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

    # exports & plots performance & losses
    pd.DataFrame.from_dict(features_track).to_csv(OUT_DIR + "features_track.csv", index=False)
    # DEBUG: print lengths of fold_v_performance before export
    print("DEBUG: fold_v_performance lengths before export:")
    for key, lst in fold_v_performance.items():
        print(f"  {key}: {len(lst)}")
    pd.DataFrame.from_dict(fold_v_performance).to_csv(OUT_DIR + "val_performance.csv", index=False)
    pd.DataFrame.from_dict(fold_test_performance).to_csv(OUT_DIR + "test_performance.csv", index=False)

