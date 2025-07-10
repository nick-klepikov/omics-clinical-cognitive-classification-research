from functions import *
from models import *
from sklearn.utils import class_weight
import networkx as nx
import yaml
from torch_geometric.utils import homophily
import argparse
import matplotlib
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd, os
import glob
import time

# ------------ Argument parsing ------------
parser = argparse.ArgumentParser(description="Train GCN on multi-omics data")
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

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)


# Set up experiment identifiers for output directory and file prefixes
model_name = args.model
modality_name = args.modality
threshold_value = args.threshold

device = check_cuda()

# Set matplotlib to 'agg'
if matplotlib.get_backend() != 'agg':
    print(f"Switching Matplotlib backend from '{matplotlib.get_backend()}' to 'agg'")
    matplotlib.use('agg')

#----------------  Main function -----------------#
if __name__ == '__main__':
    total_start = time.time()
    outer_fold_medians = {
        'outeer_fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': []
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

        # Update mastertable file and adjacency matrix path
        mastertable_file = f"{args.mastertable}_fold_{outer_fold}_thresh_{args.threshold}.csv"
        W = np.load(f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/fused_datasets/affinity_matrices/W_{args.modality}_fold_{outer_fold}_thresh_{args.threshold}.npy")

        # # Sparsify W to top-k neighbors per row (excluding self)
        # k = 10  # Fixed top-k value, can parameterize later
        # W_sparse = np.zeros_like(W)
        # for i in range(W.shape[0]):
        #     row = W[i].copy()
        #     row[i] = 0  # exclude self-loop
        #     topk_idx = np.argpartition(row, -k)[-k:]
        #     W_sparse[i, topk_idx] = W[i, topk_idx]
        # W_sparse = np.maximum(W_sparse, W_sparse.T)  # Make symmetric
        # W = W_sparse

        mastertable = pd.read_csv(mastertable_file, index_col="PATNO")

        # Filter features based on modality
        if args.modality == "geno":
            mastertable = mastertable[
                [col for col in mastertable.columns if not col.startswith("ENSG")]]
        elif args.modality == "rna":
            mastertable = mastertable[
                [col for col in mastertable.columns if
                 col.startswith("ENSG") or col in ["label", "split", "age_at_visit", "SEX_M", "EDUCATION_YEARS"]]]
        # 'fusion' keeps everything (no filtering)

        split = mastertable["split"]
        test_mask = (split == "test").to_numpy()  # Boolean mask over ALL samples
        trainval_mask = (split != "test").to_numpy()
        trainval_idx = np.where(trainval_mask)[0]  # integer indices of train+val rows

        print("Dimensions of mastertable:", mastertable.shape)  # expected (n_samples, 201)
        # extract labels and features
        y = mastertable["label"].astype(int)
        features_df = mastertable.drop(columns=["label", "split"])
        feat_names = features_df.columns.tolist()
        # prepare label dict for plotting
        labels_dict = y.to_dict()
        # feature names and numpy arrays
        features_name = features_df.columns
        y = y.to_numpy()
        X = features_df.to_numpy()
        X_indices = features_df.index

        pos = get_pos_similarity(features_df)

        # cross-validation
        folds = config["n_folds"]
        fold_v_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [], 'N_epoch': [],
                              'Loss': [], 'N_features': [], 'homophily_index': []}
        fold_test_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [],
                                 'N_epoch': []}
        features_track = {'Fold': [], 'N_selected_features': [], 'Relevant_Features': [], 'Raw_Feature_Importances': []}


        # 1) extract only train+val labels
        sub_y = y[trainval_mask]

        trainval_indices = np.where(trainval_mask)[0]

        # 2) do stratified CV on that subset
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


        for fold, (train_msk, val_msk) in enumerate(zip(raw_train, raw_val)):
            # define data splits
            X_train, X_val = X[train_msk], X[val_msk]
            y_train, y_val = y[train_msk], y[val_msk]

            # Convert to DataFrame with PATNO index and columns
            adj_df = pd.DataFrame(W, index=X_indices, columns=X_indices)

            # Use this as adjacency
            adj = adj_df

            # create graph data object
            data = create_pyg_data(adj, pd.DataFrame(data=X, columns=features_name, index=X_indices), y, train_msk, val_msk,
                                   test_mask)

            # G = nx.from_pandas_adjacency(adj)
            # # plot network
            # display_graph(
            #     fold,
            #     G,
            #     pos,
            #     labels_dict,
            #     save_fig=True,
            #     path=OUT_DIR,
            #     name_file=f"fold_{outer_fold}_{fold}_network.png",
            #     plot_title=f"Network - outer fold {outer_fold} - fold {fold}",
            #     wandb_log=False
            # )
            # create graph data object
            if "GTC_uw" in args.model:
                data.edge_attr = torch.ones((data.edge_index.shape[1], 1), device=data.edge_index.device)
            elif "GTC" in args.model or "GINE" in args.model:
                data.edge_attr = data.edge_attr.unsqueeze(-1)
            if "GPST" in args.model:
                data.x, feat_names = pad_features(data.x, config["heads"], feat_names)
            # Compute raw homophily (edge homophily)
            H_raw = homophily(data.edge_index, data.y, method='edge')
            # Compute class proportions over all nodes
            y_all = data.y.cpu().numpy()
            classes, counts = np.unique(y_all, return_counts=True)
            props = counts.astype(float) / counts.sum()
            # Baseline homophily due to class imbalance
            H_baseline = np.sum(props ** 2)
            # Normalized homophily: 0 = random, 1 = perfect homophily
            H_norm = (H_raw - H_baseline) / (1.0 - H_baseline) if (1.0 - H_baseline) > 0 else 0.0
            homophily_index = H_norm
            print(f'Normalized homophily index: {homophily_index:.4f} (raw: {H_raw:.4f}, baseline: {H_baseline:.4f})')

            # model
            model = generate_model(args.model, config, data)
            model.apply(init_weights)
            model = model.to(device)
            # compute class weights for loss function
            train_labels = data.y[data.train_mask].cpu().numpy().astype(int)

            class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.array([0, 1]),
                                                              y=train_labels)
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
            criterion.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["lrscheduler_factor"],
                                                       threshold=0.0001, patience=15,
                                                       verbose=True)
            n_epochs = config["n_epochs"]
            if "MLP" in model._get_name():
                data.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
                # training for non graph methods
                losses, performance, best_epoch, best_loss, best_model = training_mlp_nowandb(device, model, optimizer, scheduler, criterion, data, n_epochs, fold)
            else:
                # training for graph methods
                losses, performance, best_epoch, best_loss, best_model = training_nowandb(device, model, optimizer,
                                                                                scheduler, criterion, data,
                                                                                n_epochs, fold)
            # feature importance
            feat_imp_ser = feature_importance_gnnexplainer(
                model,
                data,
                names_list=feat_names,
                save_fig=True,
                name_file=f'fold_{outer_fold}_{fold}_feature_importance',
                path=OUT_DIR
            )
            # Build raw importance dict for plotting (feature->score)
            raw_imp = feat_imp_ser['Importance_score'].to_dict()
            feat_labels = feat_imp_ser.index.tolist()
            fold_v_performance, fold_test_performance, features_track = update_overall_metrics(
                fold, best_epoch, homophily_index, feat_names, feat_labels, raw_imp, performance, losses,
                fold_v_performance, fold_test_performance, features_track
            )

            # reset parameters
            # reset parameters
            for name, module in model.named_children():
                if isinstance(module, torch.nn.ModuleList):
                    for sub_module in module:
                        if hasattr(sub_module, 'reset_parameters'):
                            sub_module.reset_parameters()
                else:
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

        # exports & plots performance & losses
        #pd.DataFrame.from_dict(features_track).to_csv(OUT_DIR + f"fold_{outer_fold}_features_track.csv", index=False)

        #pd.DataFrame.from_dict(fold_test_performance).to_csv(OUT_DIR + f"fold_{outer_fold}_test_performance.csv", index=False)

            #pd.DataFrame.from_dict(fold_v_performance).to_csv(OUT_DIR + f"fold_{outer_fold}_val_performance.csv", index=False)
        # Select the inner-fold index whose test AUC is the median
        auc_list = fold_test_performance['AUC']
        median_auc = np.median(auc_list)
        # find the index of the fold whose AUC is closest to median
        idx = int(np.argmin([abs(a - median_auc) for a in auc_list]))
        print(f"Outer fold {outer_fold}: median AUC = {median_auc:.4f}, selected inner fold = {idx}")
        # retrieve the corresponding epoch and other metrics
        best_epoch = fold_v_performance['N_epoch'][idx]
        selected_metrics = {
            'Fold': outer_fold,
            'Selected_Fold': idx,
            'Epoch': best_epoch,
            'AUC': fold_test_performance['AUC'][idx],
            'Accuracy': fold_test_performance['Accuracy'][idx],
            'Recall': fold_test_performance['Recall'][idx],
            'Specificity': fold_test_performance['Specificity'][idx],
            'F1': fold_test_performance['F1'][idx]
        }
        # Debug print removed

        pd.DataFrame([selected_metrics]).to_csv(os.path.join(OUT_DIR, f"fold_{outer_fold}_selected_metrics.csv"), index=False)
        # Save selected feature importances for the median fold with Feature as column
        selected_raw_imp = features_track['Raw_Feature_Importances'][idx]
        # selected_raw_imp is a dict {feature: score}
        df_imp = pd.DataFrame.from_dict(
            selected_raw_imp, orient='index', columns=['Importance_score']
        ).reset_index().rename(columns={'index':'Feature'})
        # Save with Feature as its own column
        df_imp.to_csv(os.path.join(OUT_DIR, f"fold_{outer_fold}_selected_feature_importances.csv"), index=False)
        # Update outer_fold_medians for final aggregation
        outer_fold_medians['outer_fold'].append(outer_fold)
        outer_fold_medians['AUC'].append(selected_metrics['AUC'])
        outer_fold_medians['Accuracy'].append(selected_metrics['Accuracy'])
        outer_fold_medians['Recall'].append(selected_metrics['Recall'])
        outer_fold_medians['Specificity'].append(selected_metrics['Specificity'])
        outer_fold_medians['F1'].append(selected_metrics['F1'])

        fold_elapsed = time.time() - fold_start
        print(f"Outer fold {outer_fold} time: {fold_elapsed:.2f}s")

    # Final summary across outer folds
    summary = {}
    for m in ['AUC', 'Accuracy', 'Recall', 'Specificity', 'F1']:
        summary[f"{m}_mean"] = np.mean(outer_fold_medians[m])
        summary[f"{m}_std"] = np.std(outer_fold_medians[m])
    pd.DataFrame([summary]).to_csv(
        os.path.join(OUT_DIR, f"{model_name}_{modality_name}_thr{threshold_value}_ablation_final_summary.csv"),
        index=False
    )
    print("Final nested-CV median-based summary:", summary)

    # --- aggregate feature importances across outer folds ---
    try:
        # Read each median-fold importance CSV
        paths = glob.glob(os.path.join(OUT_DIR, "fold_*_selected_feature_importances.csv"))
        dfs = []
        for p in paths:
            df = pd.read_csv(p)
            fold = int(os.path.basename(p).split("_")[1])
            df["OuterFold"] = fold
            dfs.append(df)
        all_imp = pd.concat(dfs, ignore_index=True)

        # Count in how many outer folds each feature appears
        counts = all_imp.groupby("Feature")["OuterFold"].nunique()

        # Keep features that appear in at least 2 folds
        stable = counts[counts >= 2].index
        filtered_imp = all_imp[all_imp["Feature"].isin(stable)]

        # Compute summary: how many folds, mean & std importance
        summary_imp = (
            filtered_imp
            .groupby("Feature")["Importance_score"]
            .agg(FoldCount="nunique", MeanImportance="mean", StdImportance="std")
            .sort_values("MeanImportance", ascending=False)
            .reset_index()
        )

        # Save the final ranking
        summary_imp.to_csv(
            os.path.join(OUT_DIR, f"{model_name}_{modality_name}_thr{threshold_value}_final_feature_ranking.csv"),
            index=False
        )
        print(f"Saved final feature ranking to {model_name}_{modality_name}_thr{threshold_value}_final_feature_ranking.csv")
        print("Saved final feature ranking to final_feature_ranking.csv")
    except ValueError:
        print("Warning: No feature importance files to aggregate.")

    # Clean up intermediate per-fold CSVs
    patterns = [
        "fold_*_selected_feature_importances.csv",
        "fold_*_selected_metrics.csv"
    ]
    for pat in patterns:
        for f in glob.glob(os.path.join(OUT_DIR, pat)):
            os.remove(f)
    print("Cleaned up intermediate per-fold files.")
    total_elapsed = time.time() - total_start
    print(f"Total execution time: {total_elapsed:.2f}s")
