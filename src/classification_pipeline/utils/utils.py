# SPDX-License-Identifier: MIT
# Adapted from Gómez de Lope et al., "Graph Representation Learning Strategies for Omics Data: A Case Study on Parkinson’s Disease", arXiv:2406.14442 (MIT License)
import argparse

import torch
import pandas as pd
import numpy as np
import os

from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import networkx as nx
from torchmetrics import MetricCollection, AUROC, Accuracy, Recall, Specificity, F1Score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import copy
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from PIL import Image
import wandb
from src.classification_pipeline.models.models import *
import shap
from torch.autograd import Variable
from torch_geometric.explain import Explainer, GNNExplainer
import torch.nn.init as init
import warnings
import tarfile
import random
from imblearn.over_sampling import SMOTE

#from plot_utils import plot_from_shap_values

# --------------------------- set up------------------------------
def check_cuda():
    if torch.cuda.is_available():
        print('cuda available')
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        device = torch.device('cuda')
        torch.cuda.manual_seed(42)
    else:
        print('cuda not available')
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        device = torch.device('cpu')
    return device

# ------------------------ graph creation ------------------------
def is_symmetric(matrix: np.ndarray) -> bool:
    return np.array_equal(matrix, matrix.T)


def get_pos_similarity(X_df):
    """
    Calculate the position of nodes in a similarity graph based on cosine distance.
    Parameters:
    X_df (pandas.DataFrame): The input data_processing frame containing the data_processing points.
    Returns:
    dict: A dictionary representing the positions of the data_processing points in a graph.
    Raises:
    ValueError: If the adjacency matrix is not symmetric.
    """
    # Calculate pairwise cosine distance between data_processing points
    dist = pd.DataFrame(
        squareform(pdist(X_df, metric='cosine')),
        columns=X_df.index,
        index=X_df.index
    )
    # Calculate similarity from distance
    sim = 1 - dist
    sim = sim.fillna(0)
    sim_np = np.array(sim)
    sim_np[np.diag_indices_from(sim_np)] = 0.  # remove self-loops
    sim = pd.DataFrame(sim_np, index=sim.index, columns=sim.columns)
    if not is_symmetric(np.array(sim)):
        raise ValueError('Adjacency matrix is not symmetric')
    sim_np[sim_np < 0] = 0
    G = nx.from_pandas_adjacency(sim)
#   pos = nx.spring_layout(G, seed=42)
    pos = nx.kamada_kawai_layout(G)
    return pos


def display_graph(fold, G, pos, labels_dict=None, save_fig=False, path="./", name_file="graph.png", plot_title=None, wandb_log=True):
    """
    Draw the graph given a networkx graph G and a set of positions.
    Parameters:
    - fold (int): The fold number.
    - G (networkx.Graph): The graph object.
    - pos (dict): A dictionary mapping node IDs to positions.
    - labels_dict (dict, optional): A dictionary mapping node IDs to labels. Defaults to None.
    - save_fig (bool, optional): Whether to save the graph as an image. Defaults to False.
    - path (str, optional): The path to save the image. Defaults to "./".
    - name_file (str, optional): The name of the saved image file. Defaults to "graph.png".
    - plot_title (str, optional): The title of the graph plot. Defaults to None.
    - wandb_log (bool, optional): Whether to log the image to wandb. Defaults to True.
    """

    fig = plt.figure(figsize=(12, 12))
    weights = nx.get_edge_attributes(G, 'weight').values()
    if min(weights) == max(weights):
        # If all weights are the same, set them to a constant value (1.5)
        weights = [1.5] * len(weights)
    else:
        # Normalize weights to the range [0.5, 5]
        weights = [(weight - min(weights)) * (5 - 0.5) / (max(weights) - min(weights)) + 0.5 for weight in weights]

    if labels_dict is None:
        nx.draw(G, pos=pos, with_labels=False,
                cmap=plt.get_cmap("viridis"), node_color="blue", node_size=80,
                width=list(weights), ax=fig.add_subplot(111))
    else:
        nx.set_node_attributes(G, labels_dict, "label")
        # Get the values of the labels
        l = list(nx.get_node_attributes(G, "label").values())
        color_map = {"dodgerblue": 0, "red": 1}  # blue for controls, red for disease
        color_map = {v: k for k, v in color_map.items()}
        colors = [color_map[cat] for cat in l]
        # Get the colors of edges:
        edge_colors = []
        for u, v in G.edges():
            u_label = G.nodes[u].get("label")
            v_label = G.nodes[v].get("label")
            if u_label == v_label:
                edge_color = "gray"  # Same class edges
            else:
                edge_color = "darkorange"    # Different class edges
            edge_colors.append(edge_color)
        # Draw the graph
        nx.draw(G, pos=pos, with_labels=False,
                cmap=plt.get_cmap("viridis"), node_color=colors, node_size=80, edge_color=edge_colors,
                width=list(weights), ax=fig.add_subplot(111))
    plt.title(plot_title, fontsize=24)
    plt.tight_layout()
    if save_fig:
        fig.savefig(path + name_file)
    # Log the image to wandb: Convert the graph image to a PIL Image
    if wandb_log:
        fig.canvas.draw()  # Force the rendering of the figure
        image = Image.frombytes('RGB', fig.canvas.get_width_height(),
                            fig.canvas.tostring_rgb())
        wandb.log({f'graph-{fold}': wandb.Image(image), "caption": "Graph Visualization"})
    plt.close(fig)


def create_pyg_data(adj_df, X_df, y, train_msk, val_msk, test_msk):
    """
    Create a PyTorch Geometric Data object from the given inputs.
    Args:
        adj_df (pandas.DataFrame): The adjacency matrix as a DataFrame.
        X_df (pandas.DataFrame): The feature matrix as a DataFrame.
        y (numpy.ndarray): The target labels as a numpy array.
        train_msk (torch.Tensor): The training mask as a boolean tensor.
        val_msk (torch.Tensor): The validation mask as a boolean tensor.
        test_msk (torch.Tensor): The test mask as a boolean tensor.
    Returns:
        torch_geometric.data.Data: The PyTorch Geometric Data object.
    """
    edge_index, edge_attr = from_scipy_sparse_matrix(sp.csr_matrix(adj_df))
    data = Data(edge_index=edge_index,
                edge_attr=edge_attr,
                x=torch.tensor(X_df.values).type(torch.float),
                y=torch.tensor(y))
    data["train_mask"] = train_msk
    data["val_mask"] = val_msk
    data["test_mask"] = test_msk
    # # Gather and show some statistics about the graph.
    # print(f'Number of nodes: {data_processing.num_nodes}')
    # print(f'Number of edges: {data_processing.num_edges}')
    # print(f'Average node degree: {data_processing.num_edges / data_processing.num_nodes:.2f}')
    # print(f'Number of training nodes: {data_processing.train_mask.sum()}')
    # print(f'Training node label rate: {int(data_processing.train_mask.sum()) / data_processing.num_nodes:.2f}')
    # print(f'Has isolated nodes: {data_processing.has_isolated_nodes()}')
    # print(f'Has self-loops: {data_processing.has_self_loops()}')
    # print(f'Is undirected: {data_processing.is_undirected()}')
    # unique, counts = np.unique(data_processing.y, return_counts=True)
    # print("Classes:", unique)
    # print("Counts:", counts)
    return data

# --------- Cross-validation & metrics ----------

def init_weights(m):
    """
    Initializes the weights of a linear layer using Xavier uniform initialization.
    Args:
        m (torch.nn.Linear): The linear layer to initialize.
    """
    if isinstance(m, torch.nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)


def generate_model(model_name, config, data):
    """
    Generate a model based on the given model name.
    Args:
        model_name (str): The name of the model to generate.
        config (object): The configuration object containing the model parameters.
        n_features (int): The number of input features.
    Returns:
        model: The generated model.
    Raises:
        KeyError: If the given model name is not found in the models_dict.
    """
    if hasattr(data, 'edge_attr') and data.edge_attr is not None and len(data.edge_attr.shape) > 1:
        edge_dim = data.edge_attr.shape[1]
    else:
        edge_dim = 0
    n_features = data.num_node_features
    models_dict = { # Dictionary of model names and their corresponding lambda functions to instantiate the models only when they are actually needed.  This approach avoids initializing all models upfront.
        "MLP2": lambda: MLP2(n_features, config["cl1_hidden_units"], config["cl2_hidden_units"], config["ll_out_units"], config["dropout"]),
        "GCNN": lambda: GCNN(n_features, config["cl1_hidden_units"], config["cl2_hidden_units"], config["ll_out_units"], config["dropout"]),
        "Cheb_GCNN": lambda: Cheb_GCNN(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.K_cheby, config.ll_out_units, config.dropout),
        "GAT": lambda: GAT(n_features, config["cl1_hidden_units"], config["cl2_hidden_units"], config["heads"], config["ll_out_units"], config["dropout"]),
        "DOS_GNN": lambda: DOS_GNN(
            n_features,
            config["cl1_hidden_units"],
            config["cl2_hidden_units"],
            config["ll_out_units"],
            config["dropout"]
        ),
        "GPST": lambda: GPST(n_features, config.heads, config.ll_out_units, config.dropout, config.K_cheby),
        "GPST_GINE": lambda: GPST_GINE(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads, config.ll_out_units, config.dropout, edge_dim),
        "GPST_GINE_lin": lambda: GPST_GINE_lin(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads, config.ll_out_units, config.dropout, edge_dim),
        "GINE": lambda: GINE(n_features, config.h1_hidden_units, config.cl1_hidden_units, config.h2_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout, edge_dim),
    }
    if model_name not in models_dict:
        raise KeyError(f"Model name '{model_name}' is not found in the models_dict.")
    model = models_dict[model_name]()  # Call the lambda function to instantiate the model
    #print(model)
    return model



def update_overall_metrics(
    fold, fold_best_epoch, homophily_index,
    feat_names, relevant_features, raw_importances,
    fold_performance, fold_losses,
    dict_val_metrics, dict_test_metrics, features_track
):
    """
    Update the overall metrics with the results from a fold.
    Args:
        fold (int): The fold number.
        fold_best_epoch (int): The best epoch for the fold.
        homophily_index (float): The homophily index for the fold.
        feat_names (list): The list of feature names.
        relevant_features (list): The list of relevant features.
        fold_performance (dict): The performance metrics for each epoch in the fold.
        fold_losses (dict): The loss values for each epoch in the fold.
        dict_val_metrics (dict): The dictionary to store validation metrics.
        dict_test_metrics (dict): The dictionary to store test metrics.
        features_track (dict): The dictionary to track selected features.
    Returns:
        tuple: A tuple containing the updated dictionaries dict_val_metrics, dict_test_metrics, and features_track.
    """
    dict_val_metrics["Fold"].append(fold)
    dict_val_metrics["N_epoch"].append(fold_best_epoch)
    dict_val_metrics["N_features"].append(len(feat_names))
    dict_val_metrics["homophily_index"].append(homophily_index)

    dict_test_metrics["Fold"].append(fold)
    dict_test_metrics["N_epoch"].append(fold_best_epoch)

    features_track["Fold"].append(fold)
    features_track["N_selected_features"].append(len(feat_names))
    features_track["Relevant_Features"].append(relevant_features)
    features_track["Raw_Feature_Importances"].append(raw_importances)

    for m in fold_performance.keys():
        dict_val_metrics[m].append(fold_performance[m][fold_best_epoch][1])
        dict_test_metrics[m].append(fold_performance[m][fold_best_epoch][2])

    dict_val_metrics["Loss"].append(fold_losses[fold_best_epoch][1])
    return dict_val_metrics, dict_test_metrics, features_track

# ------------ training & evaluation ------------
def train_epoch(device, model, optimizer, criterion, data, metric):
    """Train step of model on training dataset for one epoch.
    Args:
        device (torch.device): The device to perform the training on.
        model (torch.nn.Module): The model to train.
        data (torch_geometric.data.Data): The training data_processing.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating the model parameters.
        metric (torchmetrics.Metric): The metric for evaluating the model performance.
    Returns:
        tuple: A tuple containing the training loss and the training accuracy.
    """
    model.to(device)
    model.train()
    data.to(device)
    criterion.to(device)
    optimizer.zero_grad()  # Clear gradients
    model_name = str(model.__class__.__name__)
    # Perform a single forward pass
    if ("GAT" in model_name or "GTC" in model_name or "GINE" in model_name) and not model_name == "GAT_uw": # GAT, GTC, GTC_uw, GINE, GPST_GINE_lin, GPST_GINE
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    elif "_uw" in model_name or "GPST" in model_name:  # for unweighted models exceot GTC_uw
        y_hat = model(x=data.x, edge_index=data.edge_index)
    else:
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32)) # GCN, CHEBYNET, GUNet
    # # Perform a single forward pass
    # if ("_uw" in model_name or "GPST" in model_name) and not model_name == "GTC_uw":  # for unweighted models
    #     y_hat = model(x=data_processing.x, edge_index=data_processing.edge_index)
    # elif "GAT" in model_name or "GTC" in model_name or "GINE" in model_name: # GAT, GTC, GTC_uw, GINE, GPST_GINE_lin, GPST_GINE
    #     y_hat = model(x=data_processing.x, edge_index=data_processing.edge_index, edge_attr=data_processing.edge_attr.to(torch.float32))
    # else:
    #     y_hat = model(x=data_processing.x, edge_index=data_processing.edge_index, edge_weight=data_processing.edge_attr.to(torch.float32))
    loss = criterion(y_hat[data.train_mask], data.y[data.train_mask])  # Compute the loss
    loss.backward()  # Derive gradients
    optimizer.step()  # Update parameters based on gradients
    # track loss & embeddings
    tloss = loss.detach().cpu().numpy().item()
    # track performance
    y_hat = y_hat[:,1]  # get label
    batch_acc = metric(y_hat[data.train_mask].cpu(), data.y[data.train_mask].cpu())
    train_acc = metric.compute()
    return tloss, train_acc


def evaluate_epoch(device, model, criterion, data, metric):
    """Evaluate the model on validation data_processing for a single epoch.
    Args:
        device (torch.device): The device to perform the evaluation on.
        model (torch.nn.Module): The model to evaluate.
        data (torch_geometric.data.Data): The validation data_processing.
        criterion (torch.nn.Module): The loss criterion.
        metric (torchmetrics.Metric): The evaluation metric.
    Returns:
        tuple: A tuple containing the validation loss and the validation accuracy.
    """
    model.eval()
    model.to(device)
    data.to(device)
    criterion.to(device)
    model_name = str(model.__class__.__name__)
    # Perform a single forward pass
    if ("GAT" in model_name or "GTC" in model_name or "GINE" in model_name) and not model_name == "GAT_uw": # GAT, GTC, GTC_uw, GINE, GPST_GINE_lin, GPST_GINE
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    elif "_uw" in model_name or "GPST" in model_name:  # for unweighted models exceot GTC_uw
        y_hat = model(x=data.x, edge_index=data.edge_index)
    else:
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32)) # GCN, CHEBYNET, GUNet
    if "val_mask" in data.items()._keys():
        vloss = criterion(y_hat[data.val_mask], data.y[data.val_mask])  # Compute the loss
        vloss = vloss.detach().cpu().numpy().item()
        y_hat = y_hat[:, 1]
        batch_vacc = metric(y_hat[data.val_mask].cpu(), data.y[data.val_mask].cpu())
    else:
        vloss = criterion(y_hat[data.test_mask], data.y[data.test_mask])  # Compute the loss
        vloss = vloss.detach().cpu().numpy().item()
        y_hat = y_hat[:, 1]
        batch_perf = metric(y_hat[data.test_mask].cpu(), data.y[data.test_mask].cpu())
    val_acc = metric.compute()
    return vloss, val_acc


def test_epoch(device, model, data, metric):
    """Evaluate the model on test data_processing for a single epoch.
    Args:
        device (torch.device): The device to perform the evaluation on.
        model (torch.nn.Module): The model to evaluate.
        data (torch_geometric.data.Data): The test data_processing.
        metric (torchmetrics.Metric): The metric to compute the performance.
    Returns:
        float: The test accuracy.
    """
    model.eval()
    data.to(device)
    model_name = str(model.__class__.__name__)
    # Perform a single forward pass
    if ("GAT" in model_name or "GTC" in model_name or "GINE" in model_name) and not model_name == "GAT_uw": # GAT, GTC, GTC_uw, GINE, GPST_GINE_lin, GPST_GINE
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    elif "_uw" in model_name or "GPST" in model_name:  # for unweighted models exceot GTC_uw
        y_hat = model(x=data.x, edge_index=data.edge_index)
    else:
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32)) # GCN, CHEBYNET, GUNet
    y_hat = y_hat[:, 1]  # get label
    batch_perf = metric(y_hat[data.test_mask].cpu(), data.y[data.test_mask].cpu())
    test_acc = metric.compute()
    return test_acc


def training_nowandb(device, model, optimizer, scheduler, criterion, data, n_epochs, fold):
    """Performs the full training process without logging in wandb.
    Args:
        device (torch.device): The device to be used for training.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        criterion (torch.nn.Module): The loss function used for training.
        data (torch.utils.data_processing.Dataset): The dataset used for training.
        n_epochs (int): The number of training epochs.
        fold (int): The fold number.
    Returns:
        tuple: A tuple containing the following elements:
            - losses (list): A list of training and validation losses for each epoch.
            - perf_metrics (dict): A dictionary containing performance metrics (Accuracy, AUC, Recall, Specificity, F1) for each epoch.
            - best_epoch (int): The epoch number with the best validation loss.
            - best_loss (float): The best validation loss.
            - best_model (torch.nn.Module): The model with the best validation loss.
    """
    losses = []
    #embeddings = []
    perf_metrics = {'Accuracy': [], 'AUC': [], 'Recall': [], 'Specificity': [], 'F1': []}
    train_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    val_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    test_metrics = MetricCollection({
                'Accuracy': Accuracy(task="binary"),
                'AUC': AUROC(task="binary", num_classes=2),
                'Recall': Recall(task="binary", num_classes=2),
                'Specificity': Specificity(task="binary", num_classes=2),
                'F1': F1Score(task="binary", num_classes=2),
    })
    for epoch in range(n_epochs):
        # train
        train_loss, train_perf = train_epoch(device, model, optimizer, criterion, data, train_metrics) #, epoch_embeddings
        # validation
        val_loss, val_perf = evaluate_epoch(device, model, criterion, data, val_metrics)
        # scheduler step
        scheduler.step(val_loss)
        # track losses & embeddings
        losses.append([train_loss, val_loss])
        #embeddings.append(epoch_embeddings)
        test_perf = test_epoch(device, model, data, test_metrics)
        for m in perf_metrics.keys():
                perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item(), test_perf[m].detach().numpy().item()])
        if epoch % 50 == 0:
            print(f"Epoch {epoch}",
                  f"Loss train {train_loss}",
                  f"Loss validation {val_loss}",
                  f"Acc train {train_perf}",
                  f"Acc validation {val_perf};")
        train_metrics.reset()
        val_metrics.reset()
        test_metrics.reset()

        # identify best model based on max validation AUC
        if epoch < 1:
            best_loss = losses[epoch][1]
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        else:
            if best_loss < losses[epoch][1]:
                continue
            else:
                best_loss = losses[epoch][1]
                best_model = copy.deepcopy(model)
                best_epoch = epoch
    return losses, perf_metrics, best_epoch, best_loss, best_model #, embeddings

def training_nowandb_dos_gnn(device, model, optimizer, scheduler, criterion, data, n_epochs, fold):
    """Performs the full training process without logging in wandb.
    Args:
        device (torch.device): The device to be used for training.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        criterion (torch.nn.Module): The loss function used for training.
        data (torch.utils.data_processing.Dataset): The dataset used for training.
        n_epochs (int): The number of training epochs.
        fold (int): The fold number.
    Returns:
        tuple: A tuple containing the following elements:
            - losses (list): A list of training and validation losses for each epoch.
            - perf_metrics (dict): A dictionary containing performance metrics (Accuracy, AUC, Recall, Specificity, F1) for each epoch.
            - best_epoch (int): The epoch number with the best validation loss.
            - best_loss (float): The best validation loss.
            - best_model (torch.nn.Module): The model with the best validation loss.
    """
    losses = []
    #embeddings = []
    perf_metrics = {'Accuracy': [], 'AUC': [], 'Recall': [], 'Specificity': [], 'F1': []}
    train_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    val_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    test_metrics = MetricCollection({
                'Accuracy': Accuracy(task="binary"),
                'AUC': AUROC(task="binary", num_classes=2),
                'Recall': Recall(task="binary", num_classes=2),
                'Specificity': Specificity(task="binary", num_classes=2),
                'F1': F1Score(task="binary", num_classes=2),
    })
    for epoch in range(n_epochs):
        # train
        train_loss, train_perf = train_epoch(device, model, optimizer, criterion, data, train_metrics) #, epoch_embeddings
        # validation
        val_loss, val_perf = evaluate_epoch(device, model, criterion, data, val_metrics)
        # scheduler step
        scheduler.step(val_loss)
        # track losses & embeddings
        losses.append([train_loss, val_loss])
        #embeddings.append(epoch_embeddings)
        test_perf = test_epoch(device, model, data, test_metrics)
        for m in perf_metrics.keys():
                perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item(), test_perf[m].detach().numpy().item()])
        if epoch % 50 == 0:
            print(f"Epoch {epoch}",
                  f"Loss train {train_loss}",
                  f"Loss validation {val_loss}",
                  f"Acc train {train_perf}",
                  f"Acc validation {val_perf};")
        train_metrics.reset()
        val_metrics.reset()
        test_metrics.reset()

        # identify best model based on max validation AUC
        if epoch < 1:
            best_loss = losses[epoch][1]
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        else:
            if best_loss < losses[epoch][1]:
                continue
            else:
                best_loss = losses[epoch][1]
                best_model = copy.deepcopy(model)
                best_epoch = epoch

    # === Stage 2: SMOTE on DOS-GNN embeddings + MLP head training ===
    # Prepare data_processing for SMOTE
    best_model = best_model.to(device)
    best_model.eval()
    with torch.no_grad():
        x_in     = data.x.to(device)
        edge_idx = data.edge_index.to(device)
        edge_w   = data.edge_attr.to(torch.float32).to(device)

        # First DOS-GNN layer -> produces [h1||h2]
        h = best_model.dos1(x_in, edge_idx, edge_w)
        h = F.relu(h)

        # Second DOS-GNN layer -> produces [h1'||h2']
        H = best_model.dos2(h, edge_idx, edge_w)
        H = F.relu(H)

    H = H.cpu().numpy()
    y_np = data.y.cpu().numpy()
    train_idx = data.train_mask
    val_idx   = data.val_mask
    test_idx  = data.test_mask

    # oversample only minority in embedding space
    sm = SMOTE(sampling_strategy=0.5, random_state=42)
    H_res, y_res = sm.fit_resample(H[train_idx], y_np[train_idx])

    # convert resampled data_processing to tensors
    H_res = torch.tensor(H_res, dtype=torch.float32).to(device)
    y_res = torch.tensor(y_res, dtype=torch.long).to(device)
    # prepare validation and test embeddings
    H_val  = torch.tensor(H[val_idx], dtype=torch.float32).to(device)
    y_val  = torch.tensor(y_np[val_idx], dtype=torch.long).to(device)
    H_test = torch.tensor(H[test_idx], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_np[test_idx], dtype=torch.long).to(device)

    # Build full dataset for head: train (resampled), val, test
    all_x = torch.cat([H_res, H_val, H_test], dim=0)
    all_y = torch.cat([y_res, y_val, y_test], dim=0)
    train_mask = torch.cat([
        torch.ones(H_res.size(0), dtype=torch.bool, device=all_x.device),
        torch.zeros(H_val.size(0) + H_test.size(0), dtype=torch.bool, device=all_x.device)
    ])
    val_mask = torch.cat([
        torch.zeros(H_res.size(0), dtype=torch.bool, device=all_x.device),
        torch.ones(H_val.size(0), dtype=torch.bool, device=all_x.device),
        torch.zeros(H_test.size(0), dtype=torch.bool, device=all_x.device)
    ])
    test_mask = torch.cat([
        torch.zeros(H_res.size(0) + H_val.size(0), dtype=torch.bool, device=all_x.device),
        torch.ones(H_test.size(0), dtype=torch.bool, device=all_x.device)
    ])
    head_data = Data(x=all_x, y=all_y,
                     train_mask=train_mask,
                     val_mask=val_mask,
                     test_mask=test_mask)
    # define a simple MLP head matching your existing MLP2 signature
    head = MLP2(in_f=H_res.size(1),
                h1_f=criterion.__class__.__name__ and model.lin1.in_features,  # adjust hidden sizes if needed
                h2_f=criterion.__class__.__name__ and model.lin1.out_features,
                out_f=int(data.y.max().item())+1,
                p_dropout=0.5).to(device)
    head.apply(init_weights)
    head_opt = torch.optim.Adam(head.parameters(), lr=optimizer.defaults['lr'])
    head_crit = criterion
    # train MLP head without wandb
    head_losses, head_perf, head_best_epoch, head_best_loss, best_head = \
        training_mlp_nowandb(device, head, head_opt, scheduler, head_crit,
                             head_data, n_epochs, fold)

    return losses, perf_metrics, best_epoch, best_loss, best_model, \
           head_losses, head_perf, head_best_epoch, head_best_loss, best_head


def train_mlp(device, model, optimizer, criterion, data, metric):
    """
    Trains a multi-layer perceptron (MLP) model.
    Args:
        device (torch.device): The device to perform the training on.
        model (torch.nn.Module): The MLP model to train.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss criterion used for training.
        data (torch_geometric.data.Data): The input data_processing for training.
        metric (torchmetrics.Metric): The metric used for evaluating the model performance.
    Returns:
        tuple: A tuple containing the epoch loss (float) and the training performance (float).
    """
    model.train()
    data.to(device)
    optimizer.zero_grad()  # Clear gradients.
    y_hat = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(y_hat[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    # track loss & embeddings
    epoch_loss = loss.detach().cpu().numpy().item()
    y_hat = y_hat[:,1]
    batch_perf = metric(y_hat[data.train_mask].cpu(), data.y[data.train_mask].cpu())
    train_perf = metric.compute()
    return epoch_loss, train_perf


def evaluate_mlp(device, model, criterion, data, metric):
    """
    Evaluates the performance of a multi-layer perceptron (MLP) model on the given data_processing.
    Args:
        device (torch.device): The device to perform the evaluation on.
        model (torch.nn.Module): The MLP model to evaluate.
        criterion: The loss criterion used for evaluation.
        data: The data_processing to evaluate the model on.
        metric: The performance metric used for evaluation.
    Returns:
        tuple: A tuple containing the validation loss and the validation performance.
    """
    model.eval()
    data.to(device)
    y_hat = model(data.x, data.edge_index) #_, y_hat
    if "val_mask" in data.items()._keys():
        vloss = criterion(y_hat[data.val_mask], data.y[data.val_mask])  # Compute the loss
        vloss = vloss.detach().cpu().numpy().item()
        y_hat = y_hat[:, 1]
        batch_vacc = metric(y_hat[data.val_mask].cpu(), data.y[data.val_mask].cpu())
    else:
        vloss = criterion(y_hat[data.test_mask],
                         data.y[data.test_mask])  # Compute the loss
        vloss = vloss.detach().cpu().numpy().item()
        y_hat = y_hat[:, 1]
        batch_perf = metric(y_hat[data.test_mask].cpu(), data.y[data.test_mask].cpu())

    val_perf = metric.compute()
    return vloss, val_perf


def test_mlp(device, model, data, metric):
    """
    Evaluate the performance of a multi-layer perceptron (MLP) model on test data_processing.
    Args:
        device (torch.device): The device to run the model on.
        model (torch.nn.Module): The MLP model to evaluate.
        data (torch_geometric.data.Data): The input data_processing for the model.
        metric (torch.nn.Module): The metric to compute the performance.
    Returns:
        float: The performance of the model on the test data_processing.
    """
    model.eval()
    data.to(device)
    y_hat = model(data.x, data.edge_index) # _, y_hat
    y_hat = y_hat[:,1]
    batch_perf = metric(y_hat[data.test_mask].cpu(), data.y[data.test_mask].cpu())
    test_perf = metric.compute()
    return test_perf

def training_mlp_nowandb(device, model, optimizer, scheduler, criterion, data, n_epochs, fold):
    """
    Trains a multi-layer perceptron (MLP) model without logging in wandb.
    Args:
        device (torch.device): The device to run the training on.
        model (torch.nn.Module): The MLP model to train.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        criterion (torch.nn.Module): The loss function.
        data (torch.utils.data_processing.DataLoader): The data_processing loader for training, validation, and testing data_processing.
        n_epochs (int): The number of training epochs.
        fold (int): The fold number.
    Returns:
        tuple: A tuple containing the following elements:
            - losses (list): A list of training and validation losses for each epoch.
            - perf_metrics (dict): A dictionary of performance metrics (accuracy, AUC, recall, specificity, F1) for each epoch.
            - best_epoch (int): The epoch number with the best validation loss.
            - best_loss (float): The best validation loss.
            - best_model (torch.nn.Module): The best model with the lowest validation loss.
    """
    losses = []
    perf_metrics = {'Accuracy': [], 'AUC': [], 'Recall': [], 'Specificity': [], 'F1': []}
    train_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    val_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    test_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    for epoch in range(n_epochs):
        # train
        train_loss, train_perf = train_mlp(device, model, optimizer, criterion, data, train_metrics)
        # validation
        val_loss, val_perf = evaluate_mlp(device, model, criterion, data, val_metrics)
        # scheduler step
        scheduler.step(val_loss)
        # track losses & embeddings
        losses.append([train_loss, val_loss])
        #embeddings.append(epoch_embeddings)
        test_perf = test_mlp(device, model, data, test_metrics)
        for m in perf_metrics.keys():
                perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item(), test_perf[m].detach().numpy().item()])
        if epoch % 50 == 0:
            print(f"Epoch {epoch}",
                  f"Loss train {train_loss}",
                  f"Loss validation {val_loss}",
                  f"Acc train {train_perf}",
                  f"Acc validation {val_perf};")
        train_metrics.reset()
        val_metrics.reset()
        # identify best model based on max validation AUC
        if epoch < 1:
            best_loss = losses[epoch][1]
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        else:
            if best_loss < losses[epoch][1]:
                continue
            else:
                best_loss = losses[epoch][1]
                best_model = copy.deepcopy(model)
                best_epoch = epoch
    return losses, perf_metrics, best_epoch, best_loss, best_model



# ------------ feature importance ------------

def get_feature_importance(explanation, feat_labels=None, top_k=None):
    """
    Calculates feature importance scores from the explanation object.

    Args:
        explanation: The explanation object containing feature importance information.
        feat_labels (List[str], optional): List of feature names. (default: None)
        top_k (int, optional): Number of top features to return. (default: None, returns all)

    Returns:
        pandas.DataFrame: A DataFrame containing two columns:
            - 'Feature Label': Top feature labels (as a column), set as the index.
            - 'Importance_score': Corresponding importance scores.
    """

    node_mask = explanation.get('node_mask')
    if node_mask is None:
        raise ValueError(f"The attribute 'node_mask' is not available "
                          f"in '{explanation.__class__.__name__}' "
                          f"(got {explanation.available_explanations})")
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
        raise ValueError(f"Cannot compute feature importance for "
                          f"object-level 'node_mask' "
                          f"(got shape {node_mask.size()})")

    if feat_labels is None:
        feat_labels = range(node_mask.size(1))

    score = node_mask.sum(dim=0).cpu().numpy()

    if top_k is not None:
        sorted_indices = np.argsort(score)[::-1][:top_k]
        importance_scores = score[sorted_indices]
        top_features_labels = [feat_labels[i] for i in sorted_indices]
    else:
        sorted_indices = np.argsort(score)[::-1]
        importance_scores = score[sorted_indices]
        top_features_labels = [feat_labels[i] for i in sorted_indices]

    df_top_features = pd.DataFrame({
        'Importance_score': importance_scores
    }, index= top_features_labels)
    return df_top_features


def feature_importance_gnnexplainer(model, data, names_list=None, save_fig=False, name_file='feature_importance', path=None, n=20):
    """
    Calculate the feature importance using the GNN-Explainer model.
    Args:
        model (torch.nn.Module): The GNN model.
        data (torch_geometric.data.Data): The input data_processing.
        names_list (list, optional): List of feature names. Defaults to None.
        save_fig (bool, optional): Whether to save the feature importance plot and subgraph visualization plot. Defaults to False.
        name_file (str, optional): The name of the saved files. Defaults to 'feature_importance'.
        path (str, optional): The path to save the files. Defaults to None.
        n (int, optional): The number of top features to visualize. Defaults to 20.

    Returns:
        tuple: A tuple containing the feature importance plot and the subgraph visualization plot.
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None,
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
    model_name = str(model.__class__.__name__)
    if "MLP" in model_name:  # for unweighted models
        try:
            explanation = explainer(x=data.x, edge_index=data.edge_index)
        except Exception as e:
            print(f"Catched exception: {e}")
            explanation = explainer(x=data.x, edge_index=data.edge_index)
    elif ("GAT" in model_name or "GTC" in model_name or "GINE" in model_name) and not model_name == "GAT_uw": # GAT, GTC, GTC_uw, GINE, GPST_GINE_lin, GPST_GINE
        explanation = explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    elif "_uw" in model_name or "GPST" in model_name:  # for unweighted models exceot GTC_uw
        explanation = explainer(x=data.x, edge_index=data.edge_index)
    else:
        explanation = explainer(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32)) # GCN, CHEBYNET, GUNet
    # elif ("_uw" in model_name or "GPST" in model_name) and not model_name == "GTC_uw":  # for unweighted models
    #     explanation = explainer(x=data_processing.x, edge_index=data_processing.edge_index)
    # elif "GAT" in model_name or "GTC" in model_name or "GINE" in model_name:
    #     explanation = explainer(x=data_processing.x, edge_index=data_processing.edge_index, edge_attr=data_processing.edge_attr.to(torch.float32))
    # else:
    #     explanation = explainer(x=data_processing.x, edge_index=data_processing.edge_index, edge_weight=data_processing.edge_attr.to(torch.float32))
    print(f'Generated explanations in {explanation.available_explanations}')
    if save_fig:
        if path is None:
            path = os.getcwd() + "/"
        #feat_importance = explanation.visualize_feature_importance(str(path) + name_file + ".png",
        #                                                           top_k=n, feat_labels=names_list)
        #print(f"Feature importance plot has been saved to '{path}'")
        feat_importance = get_feature_importance(explanation, names_list, top_k=None)
        #node_importance = explanation.visualize_graph(path + name_file + "_subgraph.pdf")
        #print(f"Subgraph visualization plot has been saved to '{path}'")
    else:
        #feat_importance = explanation.visualize_feature_importance(path=None,
        #                                                           top_k=n, feat_labels=names_list)
        feat_importance = get_feature_importance(explanation, names_list, top_k=None)
        #node_importance = explanation.visualize_graph(path=None)
    return feat_importance #, node_importance


def feature_importances_shap_values(model, data, X, device, names_list=None, n=20, save_fig=False, name_file='feature_importance', path=None):
    """
    Extracts the top n relevant features based on SHAP values in an ordered way
    Parameters:
        model (torch.nn.Module): The trained PyTorch model.
        data (torch.Tensor): The input data_processing for the model.
        X (pandas.DataFrame): The feature matrix.
        names_list (list, optional): The list of feature names. If not provided, the column names of X will be used.
        n (int, optional): The number of top features to extract. Default is 20.
    Returns:
        pandas.DataFrame: A DataFrame containing the top n features and their corresponding SHAP values.
    """
    # generate shap values

    # Define function to wrap model to transform data_processing to tensor
    f = lambda x: model(Variable(torch.from_numpy(x).to(device)), data.edge_index).detach().cpu().numpy()

    #explainer = shap.KernelExplainer(f, data_processing.x.cpu().detach().numpy())
    #shap_values = explainer.shap_values(data_processing.x.cpu().detach().numpy()) # takes a long time
    explainer = shap.KernelExplainer(f, shap.sample(data.x.cpu().detach().numpy(), 10))
    warnings.filterwarnings('ignore', 'The default of \'normalize\' will be set to False in version 1.2 and deprecated in version 1.4.*')
    shap_values = explainer.shap_values(data.x.cpu().detach().numpy())
    shap_values = shap_values[1]  # for binary classification_pipeline
    # convert shap values to a pandas DataFrame
    if not names_list:
        names_list = list(X.columns)
    shap_df = pd.DataFrame(shap_values, columns=names_list)
    vals = np.abs(shap_df).mean(0)
    shap_importance = pd.DataFrame(list(zip(names_list, vals)),
                                   columns=['feature', 'shap_value'])
    shap_importance.sort_values(by=['shap_value'],
                                ascending=False,
                                inplace=True)
    shap_importance = shap_importance.iloc[0:n, ]


    #plot_from_shap_values(shap_values, X, save_fig=save_fig, name_file=name_file, path=path, names_list=names_list, plot_title=None)

    return shap_importance



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory for outputs')
    parser.add_argument('--mastertable', type=str, required=True, help='Path to mastertable CSV')
    parser.add_argument('--modality', type=str, choices=["geno", "rna", "fused"], default="fused",
                        help='Modality to train on')
    parser.add_argument('--model', type=str, choices=["GCNN", "MLP2", "GAT", "DOS_GNN"], default="GCNN")
    parser.add_argument('--threshold', type=int, choices=[-2, -3, -4], required=False,
                        help='Threshold for binarizing MoCA change')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--size', type=int, default=100)


    parser.add_argument('--opt_new_G', action='store_true',
                        default=False)  # whether optimize the decoded graph based on classification result.
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--im_ratio', type=float, default=0.5)

    # Imbalance handling arguments (added for compatibility with model_runner parser)
    parser.add_argument('--setting', type=str, default='no',
                        choices=['no', 'upsampling', 'smote', 'reweight', 'embed_up', 'recon', 'newG_cls', 'recon_newG'],
                        help='Imbalance handling strategy; same as in model_runner parser')
    # upsampling: oversample in the raw input; smote: ; reweight: reweight minority classes;
    # embed_up:
    # recon: pretrain; newG_cls: pretrained decoder; recon_newG: also finetune the decoder

    parser.add_argument('--up_scale', type=float, default=1.0,
                        help='GraphSMOTE up_scale parameter: 1 for default, 0 for full balance')
    parser.add_argument('--rec_weight', type=float, default=1.0,
                        help='Reconstruction loss weight used in recon/recon_newG setting')

    print("[DEBUG] Argument list in parser:", parser._option_string_actions.keys())
    return parser

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def print_class_acc(output, labels, class_num_list, pre='valid'):
    pre_num = 0
    # print class-wise performance
    '''
    for i in range(labels.max()+1):

        cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)

        cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    # ipdb.set_trace()
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1).detach(), average='macro',
                                  multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1)[:, 1].detach(), average='macro')

    macro_F = f1_score(labels.detach(), torch.argmax(output, dim=-1).detach(), average='macro')
    print(str(pre) + ' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score, macro_F))

    return


def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    # ipdb.set_trace()
    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train


def adj_mse_loss(adj_rec, adj_tgt, adj_mask=None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2

    neg_weight = edge_num / (total_num - edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss


def load_data(mastertable, adj_path, modality, threshold, k, outer_fold):
    # output: adj, features, labels are all torch.tensor, in the dense form
    # -------------------------------------------------------

    mastertable_file = f"{mastertable}_fold_{outer_fold}_thresh_{threshold}.csv"
    W = np.load(
        f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/fused_datasets/affinity_matrices/W_{modality}_fold_{outer_fold}_thresh_{threshold}.npy")
    if k > 0:
        W_sparse = np.zeros_like(W)
        for i in range(W.shape[0]):
            row = W[i].copy()
            row[i] = 0
            topk_idx = np.argpartition(row, -k)[-k:]
            W_sparse[i, topk_idx] = W[i, topk_idx]
        W_sparse = np.maximum(W_sparse, W_sparse.T)
        W = W_sparse
    else:
        print("Skipping sparsification — using full affinity matrix")

    mastertable = pd.read_csv(mastertable_file, index_col="PATNO")

    if modality == "geno":
        mastertable = mastertable[[c for c in mastertable.columns if not c.startswith("ENSG")]]
    elif modality == "rna":
        mastertable = mastertable[
            [c for c in mastertable.columns if c.startswith("ENSG")]
            + ["label", "split", "age_at_visit", "SEX_M", "EDUCATION_YEARS"]
            ]
    # (fusion: leave all columns)

    y = mastertable["label"].astype(int).to_numpy()
    X = mastertable.drop(columns=["label", "split"]).to_numpy()

    return W, X, y
