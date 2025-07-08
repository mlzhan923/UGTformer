import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, recall_score, confusion_matrix
from model import TransformerModel, FineTunedTransformerModel
from sklearn.model_selection import KFold
from torch_geometric.data import Batch
from data import MetabolicSiteDataset
from UGTformer.lr import PolynomialDecayLR
import os.path
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
import wandb
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def gnn_collate_fn(batch):
   return Batch.from_data_list(batch)

def parse_args():
    parser = argparse.ArgumentParser(description='UGT Fine-tuning')

    parser.add_argument('--name', type=str, default='finetune',)
    parser.add_argument('--data_path', type=str, default='dataset/train/training_data.csv')
    parser.add_argument('--device', type=int, default=0, 
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=456, 
                        help='Random seed.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for cross-validation')
    

    # model parameters
    parser.add_argument('--hops', type=int, default=7,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--pe_dim', type=int, default=4,
                        help='position embedding size')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden layer size')
    parser.add_argument('--input_dim', type=int, default=132,                                                                                                                                                                                                                                      
                        help='Input dimension')
    parser.add_argument('--ffn_dim', type=int, default=256,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=5,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=6,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.4,
                        help='Dropout in the attention layer')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1400,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=300,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.00001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=5, 
                        help='Patience for early stopping')
    parser.add_argument('--save_dir', type=str, default="./", help='Directory to save models')
    parser.add_argument('--pretrained_model_path', 
                      type=str,
                      default="finetune/pretrained_model.pth",
                      help='Path to the pretrained model weights file that will be loaded before fine-tuning')
    return parser.parse_args()

args = parse_args()

if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

torch.autograd.set_detect_anomaly(True)

def train(model, optimizer, criterion, train_loader, lr_scheduler, device):
    model.train()
    total_loss = 0
    total_node_loss = 0.0
    total_graph_loss = 0.0

    node_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    graph_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    for batch in train_loader:  
        optimizer.zero_grad() 
        batch = batch.to(device)

        node_logits, graph_logits = model(batch)
        node_logits_pos = node_logits[:, 1]
        node_loss = node_criterion(node_logits_pos[batch.mask], batch.y[batch.mask].float())
    
        graph_loss = graph_criterion(graph_logits, batch.  graph_y.float())
        alpha = torch.sigmoid(model.alpha_param)
        loss = alpha * node_loss + (1 - alpha) * graph_loss
        loss.backward() 
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        total_node_loss += node_loss.item()
        total_graph_loss += graph_loss.item()
        

    avg_loss = total_loss / len(train_loader)
    avg_node_loss = total_node_loss / len(train_loader)
    avg_graph_loss = total_graph_loss / len(train_loader)
    

    return avg_loss, avg_node_loss, avg_graph_loss


def evaluate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    total_node_loss = 0.0
    total_graph_loss = 0.0
    all_node_probs, all_node_preds, all_node_labels = [], [], []
    all_graph_probs, all_graph_preds, all_graph_labels = [], [], []

    alpha = torch.sigmoid(model.alpha_param)

    node_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    graph_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            node_logits, graph_logits = model(batch)
            node_logits_pos = node_logits[:, 1]

            node_loss = node_criterion(node_logits_pos[batch.mask], batch.y[batch.mask].float())
            graph_loss = graph_criterion(graph_logits, batch.graph_y.float())
            loss = alpha * node_loss + (1 - alpha) * graph_loss            
            total_loss += loss.item()
            total_node_loss += node_loss.item()
            total_graph_loss += graph_loss.item()

            node_probs_np = torch.sigmoid(node_logits_pos)[batch.mask].cpu().numpy()
            node_preds = (node_probs_np > 0.5).astype(int)
            all_node_probs.extend(node_probs_np)
            all_node_preds.extend(node_preds)
            all_node_labels.extend(batch.y[batch.mask].cpu().numpy())

            graph_probs = torch.sigmoid(graph_logits).cpu().numpy()
            graph_preds = (graph_probs > 0.5).astype(int)
            all_graph_probs.extend(graph_probs)
            all_graph_preds.extend(graph_preds)
            all_graph_labels.extend(batch.graph_y.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    avg_node_loss = total_node_loss / len(val_loader)
    avg_graph_loss = total_graph_loss / len(val_loader)

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

    return (avg_loss, avg_node_loss, avg_graph_loss, graph_auroc, graph_acc, graph_f1, graph_mcc, graph_sensitivity, graph_specificity, 
        node_auroc, node_acc, node_f1, node_mcc, node_sensitivity, node_specificity,
       )

    
def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    smiles_list, labels = MetabolicSiteDataset.load_dataset(args.data_path)
    dataset = MetabolicSiteDataset(root='./', smiles_list=smiles_list, labels=labels)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    
    fold_metrics = []

    t_total = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        assert len(set(train_idx).intersection(set(val_idx))) == 0, "Train and validation sets overlap!"

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=gnn_collate_fn)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=gnn_collate_fn)

        pretrained_model = TransformerModel(
            hops=args.hops,
            n_class=2,
            input_dim=args.input_dim,
            pe_dim=args.pe_dim,
            emb_dim=args.hidden_dim,
            n_layers=args.n_layers,
            num_heads=args.n_heads,
            hidden_dim=args.hidden_dim,
            ffn_dim=args.ffn_dim,
            dropout_rate=args.dropout,
            attention_dropout_rate=args.attention_dropout
        ).to(device)

        pretrained_model.load_state_dict(torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage.cuda(args.device)))
        
        model = FineTunedTransformerModel(
            pretrained_model=pretrained_model,  
            hidden_dim=args.hidden_dim,            
            num_classes=2                      
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)

        lr_scheduler = PolynomialDecayLR(
            optimizer=optimizer,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0
        )

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        best_loss = float('inf')  
        early_stopping_counter = 0  
        early_stopping_iter = args.patience  
        best_fold_metrics = {}  
        model_save_path = os.path.join(args.save_dir, f"best_finetuned_model_fold_{fold_idx + 1}.pth")

        for epoch in range(args.epochs):
            train_loss, train_node_loss, train_graph_loss = train(model, optimizer, criterion, train_loader, lr_scheduler, device)
            alpha = torch.sigmoid(model.alpha_param)
            valid_loss, valid_node_loss, valid_graph_loss, graph_auroc, graph_acc, graph_f1, graph_mcc, graph_se, graph_sp, node_auroc, node_acc, node_f1, node_mcc, node_se, node_sp = evaluate(
                model, criterion, val_loader, device
            )
   
            print(f"Fold {fold_idx + 1}, Epoch {epoch + 1}: "
                  f"Train Loss: {train_loss:.3f},"
                  f"Train Node Loss: {train_node_loss:.3f},"
                  f"Train Graph Loss: {train_graph_loss:.3f},"
                  f"Valid Loss: {valid_loss:.3f},"
                  f"Valid Node Loss: {valid_node_loss:.3f},"
                  f"Valid Graph Loss: {valid_graph_loss:.3f},"
                  f"Node AUROC: {node_auroc:.4f}, Graph AUROC: {graph_auroc:.3f}")

            if (epoch + 1) % 3 == 0:
                avg_valid_loss = valid_loss  
                
                if avg_valid_loss < best_loss:
                    best_loss = avg_valid_loss  
                    early_stopping_counter = 0  
                    torch.save(model.state_dict(), model_save_path)

                    best_fold_metrics = {
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
                    }
                else:
                    early_stopping_counter += 1  

                if early_stopping_counter >= early_stopping_iter:
                    print(f"Early stopping after {early_stopping_counter} epochs without improvement")
                    break

        fold_metrics.append(best_fold_metrics)

    print("Train cost: {:.3f}s".format(time.time() - t_total))

    avg_metrics = {key: np.mean([metric[key] for metric in fold_metrics]) for key in fold_metrics[0]}
    std_metrics = {key: np.std([metric[key] for metric in fold_metrics]) for key in fold_metrics[0]}

    print("5-Fold Average Metrics:")
    for key in avg_metrics:
        print(f"{key}: {avg_metrics[key]:.3f} ± {std_metrics[key]:.3f}")

if __name__ == '__main__':
    main()



