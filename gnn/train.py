from config import args
from loader import MetabolicSiteDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from model import GNNClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, recall_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader,Subset
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Batch 
import random
import wandb


def gnn_collate_fn(batch):
    return Batch.from_data_list(batch)

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_node_loss, total_graph_loss = 0, 0
    correct_node, correct_graph, total_node, total_graph = 0, 0, 0, 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        node_probs, graph_prob = model(data)
        
        node_labels = data.y[data.mask].float()
        node_loss = criterion(node_probs[data.mask], node_labels)

        graph_labels = data.graph_y.float()
        graph_loss = criterion(graph_prob, graph_labels)  

        alpha = torch.sigmoid(model.alpha_param)
        loss = alpha * node_loss + (1 - alpha) * graph_loss
        loss.backward()
        optimizer.step()

        total_node_loss += node_loss.item()
        total_graph_loss += graph_loss.item()

        node_preds = (node_probs[data.mask] > 0.5).int()
        correct_node += (node_preds == node_labels.int()).sum().item()
        total_node += node_labels.size(0)

        graph_preds = (graph_prob > 0.5).int()  
        correct_graph += (graph_preds == graph_labels.int()).sum().item()
        total_graph += graph_labels.size(0)

    avg_node_loss = total_node_loss / len(train_loader)
    avg_graph_loss = total_graph_loss / len(train_loader)
    node_accuracy = correct_node / total_node
    graph_accuracy = correct_graph / total_graph

    return avg_node_loss, avg_graph_loss, node_accuracy, graph_accuracy


def evaluate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    total_node_loss, total_graph_loss = 0, 0
    all_node_probs, all_node_preds, all_node_labels = [], [], []
    all_graph_probs, all_graph_preds, all_graph_labels = [], [], []

    alpha = torch.sigmoid(model.alpha_param)

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)                          
            node_probs, graph_prob = model(data)

            node_labels = data.y[data.mask].float() 
            node_loss = criterion(node_probs[data.mask], node_labels)
            total_node_loss += node_loss.item()

            node_probs_np = node_probs[data.mask].cpu().numpy()
            node_preds = (node_probs_np > 0.5).astype(int)
            all_node_probs.extend(node_probs_np)
            all_node_preds.extend(node_preds)
            all_node_labels.extend(node_labels.cpu().numpy())

            graph_labels = data.graph_y.float()
            graph_loss = criterion(graph_prob, graph_labels)
            total_graph_loss += graph_loss.item()

            graph_probs = graph_prob.cpu().numpy()
            graph_preds = (graph_probs > 0.5).astype(int)
            all_graph_probs.extend(graph_probs)
            all_graph_preds.extend(graph_preds)
            all_graph_labels.extend(graph_labels.cpu().numpy())

            loss = alpha * node_loss + (1 - alpha) * graph_loss
            total_loss += loss.item()

    node_auroc = roc_auc_score(all_node_labels, all_node_probs)
    node_acc = accuracy_score(all_node_labels, all_node_preds)
    node_f1 = f1_score(all_node_labels, all_node_preds)
    node_mcc = matthews_corrcoef(all_node_labels, all_node_preds)
    node_sensitivity = recall_score(all_node_labels, all_node_preds)  

    tn, fp, fn, tp = confusion_matrix(all_node_labels, all_node_preds).ravel()
    node_specificity = tn / (tn + fp)

    graph_auroc = roc_auc_score(all_graph_labels, all_graph_probs)
    graph_acc = accuracy_score(all_graph_labels, all_graph_preds)
    graph_f1 = f1_score(all_graph_labels, all_graph_preds)
    graph_mcc = matthews_corrcoef(all_graph_labels, all_graph_preds)
    graph_sensitivity = recall_score(all_graph_labels, all_graph_preds)


    tn, fp, fn, tp = confusion_matrix(all_graph_labels, all_graph_preds).ravel()
    graph_specificity = tn / (tn + fp)

    avg_loss = total_loss / len(val_loader)
    avg_node_loss = total_node_loss / len(val_loader)
    avg_graph_loss = total_graph_loss / len(val_loader)

    return (avg_loss, avg_node_loss, avg_graph_loss,
            node_auroc, node_acc, node_f1, node_mcc, node_sensitivity, node_specificity,
            graph_auroc, graph_acc, graph_f1, graph_mcc, graph_sensitivity, graph_specificity)


def run_training(train_loader, valid_loader, model_type, model_save_path, device):
   
    model = GNNClassifier(
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=model_type
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCELoss()  

    best_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_iter = 10

    for epoch in range(args.epochs):
        train_node_loss, train_graph_loss, train_node_acc, train_graph_acc = train(model, optimizer, criterion, train_loader, device)
        alpha = torch.sigmoid(model.alpha_param)
        valid_loss = evaluate(model, criterion, valid_loader, device)[0]

        print(f"Epoch {epoch + 1}, Train Node Loss: {train_node_loss:.3f}, Train Graph Loss: {train_graph_loss:.3f}, "
              f"Valid Loss: {valid_loss:.3f},")

        avg_valid_loss = valid_loss
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_iter:
                print("Early stopping...")
                break

    return best_loss

def run_testing(valid_loader, model_type, model_save_path, device):
    model = GNNClassifier(
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=model_type
    )
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)

    criterion = torch.nn.BCELoss()  
    valid_metrics = evaluate(model, criterion, valid_loader, device)
    avg_loss = valid_metrics[0]
    node_auroc, node_acc, node_f1, node_mcc, node_se, node_sp = valid_metrics[3:9]
    graph_auroc, graph_acc, graph_f1, graph_mcc, graph_se, graph_sp = valid_metrics[9:]
    
    return avg_loss, graph_auroc, graph_acc, graph_f1, graph_mcc, graph_se, graph_sp, node_auroc, node_acc, node_f1, node_mcc, node_se, node_sp


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_random_seed(0)
    print("=============== Args ===============")
    print(args)
    print("=====================================")
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    file_path = 'dataset/train/training_data.csv'
    smiles_list, labels = MetabolicSiteDataset.load_dataset(file_path)  
    dataset = MetabolicSiteDataset(root='./', smiles_list=smiles_list, labels=labels)  
    
    model_type = args.gnn_type

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_save_dir = os.path.join(args.save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)


    kf = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, collate_fn=gnn_collate_fn, shuffle=True)
        valid_loader = DataLoader(val_subset, batch_size=args.batch_size, collate_fn=gnn_collate_fn, shuffle=False)

        model_save_path = os.path.join(model_save_dir, f'{model_type}_fold_{fold_idx + 1}.pt')
        run_training(train_loader, valid_loader, model_type, model_save_path, device)

        valid_loss, graph_auroc, graph_acc, graph_f1, graph_mcc, graph_se, graph_sp, node_auroc, node_acc, node_f1, node_mcc, node_se, node_sp = \
            run_testing(valid_loader, model_type, model_save_path, device)

        fold_metrics.append({
            'valid_loss': valid_loss,
            'graph_auroc': graph_auroc,
            'graph_accuracy': graph_acc,
            'graph_f1': graph_f1,
            'graph_mcc': graph_mcc,
            'graph_sensitivity': graph_se,
            'graph_specificity': graph_sp,
            'node_auroc': node_auroc,
            'node_accuracy': node_acc,
            'node_f1': node_f1,
            'node_mcc': node_mcc,
            'node_sensitivity': node_se,
            'node_specificity': node_sp
        })

    avg_metrics = {key: np.mean([metric[key] for metric in fold_metrics]) for key in fold_metrics[0]}
    std_metrics = {key: np.std([metric[key] for metric in fold_metrics]) for key in fold_metrics[0]}
    
    print(f"Average metrics for {model_type}:")
    for key in avg_metrics:
        print(f"{key}: {avg_metrics[key]:.3f} ± {std_metrics[key]:.3f}")


if __name__ == "__main__":
    main()