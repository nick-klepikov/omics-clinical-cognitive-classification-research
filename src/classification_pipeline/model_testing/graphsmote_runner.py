import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score, roc_auc_score
import yaml

import torch
import torch.nn.functional as F
import torch.optim as optim

import src.classification_pipeline.models.models as models
import src.classification_pipeline.utils.utils as utils
import random

import copy
class_num_mat = None

USE_THRESHOLD_TUNING = True  # set to False to skip tuning (use 0.5)



# Helper to (re-)initialize model weights for each CV fold
def init_weights(m):
    # If the module has a reset_parameters method, call it
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

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


def train(epoch):
    W_new = W  # default to original adjacency
    loss_rec = torch.tensor(0.0, device=W.device)
    y_new = y
    idx_train_new = idx_train
    encoder.train()
    classifier.train()
    decoder.train()
    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    embed = encoder(X, W)

    if args.setting == 'recon_newG' or args.setting == 'recon' or args.setting == 'newG_cls':
        ori_num = y.shape[0]
        embed, y_new, idx_train_new, W_up = utils.recon_upsample(embed, y, idx_train, adj=W.detach().to_dense(),
                                                                 portion=args.up_scale, im_class_num=im_class_num)
        generated_G = decoder(embed)

        loss_rec = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], W.detach().to_dense())

        if not args.opt_new_G:
            W_new = copy.deepcopy(generated_G.detach())
            threshold = 0.5
            W_new[W_new < threshold] = 0.0
            W_new[W_new >= threshold] = 1.0

            edge_ac = W_new[:ori_num, :ori_num].eq(W.to_dense()).double().sum() / (ori_num ** 2)
        else:
            W_new = generated_G
            edge_ac = F.l1_loss(W_new[:ori_num, :ori_num], W.to_dense(), reduction='mean')

        # calculate generation information
        exist_edge_prob = W_new[:ori_num, :ori_num].mean()  # edge prob for existing nodes
        generated_edge_prob = W_new[ori_num:, :ori_num].mean()  # edge prob for generated nodes

        W_new = torch.mul(W_up, W_new)

        exist_edge_prob = W_new[:ori_num, :ori_num].mean()  # edge prob for existing nodes
        generated_edge_prob = W_new[ori_num:, :ori_num].mean()  # edge prob for generated nodes

        W_new[:ori_num, :][:, :ori_num] = W.detach().to_dense()

        if not args.opt_new_G:
            W_new = W_new.detach()

        if args.setting == 'newG_cls':
            idx_train_new = idx_train
    elif args.setting == 'embed_up':
        # perform SMOTE in embedding space
        embed, labels_new, idx_train_new = utils.recon_upsample(embed, y, idx_train, portion=args.up_scale,
                                                                im_class_num=im_class_num)
        W_new = W
    else:
        y_new = y
        idx_train_new = idx_train
        W_new = W
    output = classifier(embed, W_new)

    loss_train = F.cross_entropy(output[idx_train_new], y_new[idx_train_new])

    acc_train = utils.accuracy(output[idx_train], y_new[idx_train])

    if args.setting == 'recon_newG':
        loss = loss_train + loss_rec * args.rec_weight
    elif args.setting == 'recon':
        loss = loss_rec + 0 * loss_train
    else:
        loss = loss_train
        loss_rec = loss_train

    loss.backward()

    if args.setting == 'newG_cls':
        optimizer_en.zero_grad()
        optimizer_de.zero_grad()
    else:
        optimizer_en.step()

    optimizer_cls.step()

    if args.setting == 'recon_newG' or args.setting == 'recon':
        optimizer_de.step()

    loss_val = F.cross_entropy(output[idx_val], y[idx_val])
    acc_val = utils.accuracy(output[idx_val], y[idx_val])

    utils.print_class_acc(output[idx_val], y[idx_val], class_num_mat[:, 1])

    print('Epoch: {:05d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_rec: {:.4f}'.format(loss_rec.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))

    # Return metrics for logging
    return loss_train.item(), loss_rec.item(), loss_val.item(), acc_val.item()


def test(epoch=0):
    if class_num_mat is None:
        raise RuntimeError("class_num_mat must be initialized before calling test()")
    encoder.eval()
    classifier.eval()
    decoder.eval()
    embed = encoder(X, W)
    output = classifier(embed, W)
    loss_test = F.cross_entropy(output[idx_test], y[idx_test])
    acc_test = utils.accuracy(output[idx_test], y[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    utils.print_class_acc(output[idx_test], y[idx_test], class_num_mat[:, 2], pre='test')


def save_model(epoch):
    saved_content = {}

    saved_content['encoder'] = encoder.state_dict()
    saved_content['decoder'] = decoder.state_dict()
    saved_content['classifier'] = classifier.state_dict()

    torch.save(saved_content,
               'checkpoint/{}/{}_{}_{}_{}.pth'.format(args.dataset, args.setting, epoch, args.opt_new_G, args.im_ratio))

    return


def load_model(filename):
    loaded_content = torch.load('checkpoint/{}/{}.pth'.format(args.dataset, filename),
                                map_location=lambda storage, loc: storage)

    encoder.load_state_dict(loaded_content['encoder'])
    decoder.load_state_dict(loaded_content['decoder'])
    classifier.load_state_dict(loaded_content['classifier'])

    print("successfully loaded: " + filename)

    return


if __name__ == '__main__':
    # Training setting
    parser = utils.get_parser()

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    # Load YAML hyperparameters
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    outer_fold_results = []
    n_outer = int(config.get("n_folds", 5))
    for outer_fold in range(n_outer):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        print(f"=== Outer fold {outer_fold} ===")
        # Build file paths for this fold
        mastertable_file = f"{args.mastertable}_fold_{outer_fold}_thresh_{args.threshold}.csv"
        adj_path = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/intermid/fused_datasets/affinity_matrices/W_{args.modality}_fold_{outer_fold}_thresh_{args.threshold}.npy"

        # Load and preprocess data for this fold
        W, X, y = utils.load_data(
            mastertable=args.mastertable,
            adj_path=adj_path,
            modality=args.modality,
            threshold=args.threshold,
            k=50,
            outer_fold=outer_fold
        )
        im_class_num = 1

        # Move loaded arrays to torch tensors on the correct device
        device = torch.device('cuda' if args.cuda else 'cpu')
        W = torch.from_numpy(W).float().to(device)
        X = torch.from_numpy(X).float().to(device)
        y = torch.from_numpy(y).long().to(device)

        # Split into train+val / test by 'split' column in mastertable
        df = pd.read_csv(mastertable_file, index_col="PATNO")
        split = df["split"].to_numpy()
        test_mask = (split == "test")
        trainval_mask = (split != "test")
        trainval_idx = np.where(trainval_mask)[0]

        # Inner CV on train+val to build candidate masks
        sub_y = y[trainval_mask]
        trainval_indices = trainval_idx
        n_inner = int(config.get("n_folds", 5))
        skf_inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=args.seed)
        raw_train_masks = []
        raw_val_masks = []

        for rel_tr, rel_va in skf_inner.split(np.zeros(len(sub_y)), sub_y):
            mask_tr = torch.tensor(
                np.isin(np.arange(len(y)), trainval_indices[rel_tr]),
                dtype=torch.bool, device=X.device
            )
            mask_va = torch.tensor(
                np.isin(np.arange(len(y)), trainval_indices[rel_va]),
                dtype=torch.bool, device=X.device
            )
            raw_train_masks.append(mask_tr)
            raw_val_masks.append(mask_va)

        # ---------- Inner 5-fold nested CV (dense adjacency) ----------
        # Only store AUCs for inner-fold selection
        fold_test_aucs = []
        fold_test_accs = []
        fold_test_recalls = []
        fold_test_specificities = []
        fold_test_f1s = []
        fold_best_epochs = []
        inner_thresholds = []

        for ifold, (mask_tr, mask_va) in enumerate(zip(raw_train_masks, raw_val_masks)):
            print(f"--- Inner fold {ifold} ---")
            # start timer for this inner fold
            t_inner_start = time.time()
            # Compute class_num_mat for this inner fold (train/val/test counts)
            # Convert masks to index tensors
            idx_tr_tensor = mask_tr.nonzero(as_tuple=True)[0]
            idx_va_tensor = mask_va.nonzero(as_tuple=True)[0]
            idx_te_tensor = torch.tensor(test_mask, dtype=torch.bool, device=X.device).nonzero(as_tuple=True)[0]
            num_classes = int(y.max().item()) + 1
            class_counts = torch.zeros((num_classes, 3), dtype=torch.long, device=X.device)
            for c in range(num_classes):
                class_counts[c, 0] = int((y[idx_tr_tensor] == c).sum())
                class_counts[c, 1] = int((y[idx_va_tensor] == c).sum())
                class_counts[c, 2] = int((y[idx_te_tensor] == c).sum())
            # Set global class_num_mat for use in train()
            class_num_mat = class_counts


            # 1) Build fresh models & reset weights per fold
            if args.setting != "embed_up":
                encoder = models.GCN_En(nfeat=X.shape[1], nhid=args.nhid, nembed=args.nhid, dropout=float(config["dropout"]))
                classifier = models.GCN_Classifier(nembed=args.nhid, nhid=args.nhid, nclass=int(y.max().item()) + 1,
                                                   dropout=float(config["dropout"]))
            else:
                encoder = models.GCN_En2(nfeat=X.shape[1], nhid=args.nhid, nembed=args.nhid,
                                        dropout=float(config["dropout"]))
                classifier = models.GCN_Classifier(nembed=args.nhid, nhid=args.nhid, nclass=int(y.max().item()) + 1,
                                                   dropout=float(config["dropout"]))
            decoder = models.Decoder(nembed=args.nhid, dropout=float(config["dropout"]))
            if args.cuda:
                encoder, classifier, decoder = encoder.cuda(), classifier.cuda(), decoder.cuda()
            encoder.apply(init_weights)
            classifier.apply(init_weights)
            decoder.apply(init_weights)

            # 2) New optimizers & scheduler (YAML-driven)
            optimizer_en = optim.Adam(encoder.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
            optimizer_cls = optim.Adam(classifier.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
            optimizer_de = optim.Adam(decoder.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
            sched_inner = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_en, mode='min',
                factor=float(config["lrscheduler_factor"]),
                patience=20,
                verbose=False
            )

            # 3) Inner‐loop training & early‐stop by val loss
            best_val_loss = float('inf')
            best_epoch = 0
            for ep in range(int(config["n_epochs"])):
                # Convert boolean masks to index tensors
                idx_train = mask_tr.nonzero(as_tuple=True)[0]
                idx_val = mask_va.nonzero(as_tuple=True)[0]
                loss_tr, loss_rec, loss_val, acc_val = train(ep)
                sched_inner.step(loss_val)
                if loss_val < best_val_loss:
                    best_val_loss, best_epoch = loss_val, ep
            # report inner-fold training time
            print(f"Inner fold {ifold} training time: {time.time() - t_inner_start:.2f}s")
            try:
                fold_best_epochs.append(best_epoch)
            except NameError:
                fold_best_epochs = [best_epoch]

                # --- 3.5) Tune threshold on this inner fold’s validation set ---
            if USE_THRESHOLD_TUNING:
                encoder.eval()
                classifier.eval()
                with torch.no_grad():
                    embed = encoder(X, W)
                    logits = classifier(embed, W)
                    # validation mask tensor
                    val_mask_tensor = mask_va
                    # compute validation probabilities and labels
                    val_probs = torch.softmax(logits[val_mask_tensor], dim=1)[:, 1].cpu().numpy()
                    val_labels = y[val_mask_tensor].cpu().numpy()
                    # find best threshold for max F1 on validation
                    best_thresh, best_f1_val = find_best_threshold(val_probs, val_labels, metric=f1_score)
                    inner_thresholds.append(best_thresh)
                    print(f"[Threshold Tuning] fold {ifold}: threshold={best_thresh:.2f}, F1_val={best_f1_val:.3f}")
            else:
                inner_thresholds.append(0.5)

            # 4) Evaluate this fold on the outer test set
            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                if USE_THRESHOLD_TUNING:
                    median_thresh = float(np.median(inner_thresholds))
                else:
                    median_thresh = 0.5
                print(f"[Test Eval] using median threshold: {median_thresh:.2f}")
                embed = encoder(X, W)
                logits = classifier(embed, W)
                idx_test = torch.tensor(test_mask, dtype=torch.bool, device=X.device)
                probs = torch.softmax(logits[idx_test], dim=1)[:, 1].cpu().numpy()
                truths = y[idx_test].cpu().numpy()
                preds = (torch.softmax(logits[idx_test], dim=1)[:, 1].cpu().numpy() >= median_thresh).astype(int)

                auc = roc_auc_score(truths, probs)
                # Store all metrics for inner-fold selection
                fold_test_aucs.append(auc)
                fold_test_accs.append(accuracy_score(truths, preds))
                fold_test_recalls.append(recall_score(truths, preds, zero_division=0))
                cm = confusion_matrix(truths, preds)
                spec = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0.0
                fold_test_specificities.append(spec)
                fold_test_f1s.append(f1_score(truths, preds, zero_division=0))

        # 5) Select the fold whose test-AUC is closest to the median
        med_auc = np.median(fold_test_aucs)
        best_if = int(np.argmin([abs(a - med_auc) for a in fold_test_aucs]))
        print(f"Selected inner fold {best_if} with AUC={fold_test_aucs[best_if]:.4f}")
        selected_acc = fold_test_accs[best_if]
        selected_rec = fold_test_recalls[best_if]
        selected_spec = fold_test_specificities[best_if]
        selected_f1 = fold_test_f1s[best_if]

        # Override train/val indices for the outer run
        idx_train = raw_train_masks[best_if].nonzero(as_tuple=True)[0]
        idx_val   = raw_val_masks[best_if].nonzero(as_tuple=True)[0]
        idx_test  = torch.tensor(test_mask, dtype=torch.bool, device=X.device)
        # Store all metrics for this outer fold
        outer_fold_results.append({
            'outer_fold': outer_fold,
            'AUC': fold_test_aucs[best_if],
            'Accuracy': selected_acc,
            'Recall': selected_rec,
            'Specificity': selected_spec,
            'F1': selected_f1
        })

    # ---------- Summary of outer CV ----------
    df_outer = pd.DataFrame(outer_fold_results)
    print(df_outer.agg(['mean','std']))

    # ----- Final summary save -----
    summary = {}
    for m in ['AUC', 'Accuracy', 'Recall', 'Specificity', 'F1']:
        summary[f'{m}_mean'] = df_outer[m].mean()
        summary[f'{m}_std'] = df_outer[m].std()
    summary_df = pd.DataFrame([summary])
    summary_filename = f"/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/ablation_experiments/GCNN_{args.modality}_thr{args.threshold}_final_summary.csv"
    summary_df.to_csv(summary_filename, index=False)