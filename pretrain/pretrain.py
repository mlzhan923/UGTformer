import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import random
import os
import matplotlib.pyplot as plt
import rdkit
from tqdm import tqdm
import numpy as np
from lr import PolynomialDecayLR
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool  
from model import TransformerModel
from decoder import Model_decoder 
from data import MoleculeDataset, molgraph_to_graph_data
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message="scatter_reduce_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms"
)

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL) 

def collate_fn(batch):
    return batch


def train(model_list, loader, optimizer_list, lr_scheduler, device):
    model, model_decoder = model_list
    model.train()
    model_decoder.train()
    if_auc, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0

    total_loss = 0  
    num_batches = 0  

    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=True)):       
        graph_batch = molgraph_to_graph_data(batch)  
        graph_batch = graph_batch.to(device)

        node_emb, graph_emb = model(graph_batch)         
        loss, bond_if_auc, bond_type_acc, atom_type_acc, atom_num_rmse, bond_num_rmse = model_decoder(batch, node_emb, graph_emb, graph_batch.batch)
        
        optimizer_list.zero_grad()
        loss.backward()
        optimizer_list.step()
        lr_scheduler.step()

        if_auc += bond_if_auc

        type_acc += bond_type_acc
        a_type_acc += atom_type_acc
        a_num_rmse += atom_num_rmse
        b_num_rmse += bond_num_rmse

        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_if_auc = if_auc / num_batches if num_batches > 0 else 0
    avg_bond_type_acc = type_acc / num_batches if num_batches > 0 else 0
    avg_atom_type_acc = a_type_acc / num_batches if num_batches > 0 else 0
    avg_a_num_rmse = a_num_rmse / num_batches if num_batches > 0 else 0
    avg_b_num_rmse = b_num_rmse / num_batches if num_batches > 0 else 0

    return avg_loss, avg_if_auc, avg_bond_type_acc, avg_atom_type_acc, avg_a_num_rmse, avg_b_num_rmse
    
def validate(model_list, loader, device):
    model, model_decoder = model_list
    model.eval()
    model_decoder.eval()
    total_loss = 0  
    num_batches = 0  
    if_auc, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", disable=True):
            graph_batch = molgraph_to_graph_data(batch).to(device)
            node_emb, graph_emb = model(graph_batch)
            loss, bond_if_auc, bond_type_acc, atom_type_acc, atom_num_rmse, bond_num_rmse = model_decoder(
                batch, node_emb, graph_emb, graph_batch.batch
            )
            total_loss += loss.item()
            if_auc += bond_if_auc
            type_acc += bond_type_acc
            a_type_acc += atom_type_acc
            a_num_rmse += atom_num_rmse
            b_num_rmse += bond_num_rmse
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_if_auc = if_auc / num_batches if num_batches > 0 else 0
    avg_bond_type_acc = type_acc / num_batches if num_batches > 0 else 0
    avg_atom_type_acc = a_type_acc / num_batches if num_batches > 0 else 0
    avg_a_num_rmse = a_num_rmse / num_batches if num_batches > 0 else 0
    avg_b_num_rmse = b_num_rmse / num_batches if num_batches > 0 else 0

    return avg_loss, avg_if_auc, avg_bond_type_acc, avg_atom_type_acc, avg_a_num_rmse, avg_b_num_rmse

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
 
    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
 
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)') 
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 10)')
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

    parser.add_argument('--dataset', type=str, default='pretrain/all.txt', 
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    
    
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
    
    parser.add_argument('--output_model_file', type=str, default='pretrain/saved_model/',
                        help='filename to output the pre-trained model')
    parser.add_argument('--graph_dir', type=str, default='pretrain/graph',
                        help='Directory to save graph outputs')


    args = parser.parse_args()
    set_seed(0) 

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    # 创建保存模型的目录
    output_dir = os.path.dirname(args.output_model_file)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    
    out_dir = args.graph_dir
    os.makedirs(out_dir, exist_ok=True)

    dataset = MoleculeDataset(args.dataset)

    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TransformerModel(
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

    model_decoder = Model_decoder(args.hidden_dim, device).to(device)

    model_list = [model, model_decoder]
    optimizer = optim.Adam([{"params":model.parameters()},{"params":model_decoder.parameters()}], lr=args.peak_lr, weight_decay=args.weight_decay)

    lr_scheduler = PolynomialDecayLR(
            optimizer=optimizer,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0
        )

    avg_train_losses = []  
    avg_val_losses = []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_if_auc, train_bond_type_acc, train_atom_type_acc, train_a_num_rmse, train_b_num_rmse = train(
            model_list, train_loader, optimizer, lr_scheduler, device
        )
        avg_train_losses.append(train_loss)

        val_loss, val_if_auc, val_bond_type_acc, val_atom_type_acc, val_a_num_rmse, val_b_num_rmse = validate(
            model_list, val_loader, device
        )
        avg_val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train -> Loss: {train_loss:.4f}, IF AUC: {train_if_auc:.4f}, Bond Type Acc: {train_bond_type_acc:.4f}, "
              f"Atom Type Acc: {train_atom_type_acc:.4f}, Atom Num RMSE: {train_a_num_rmse:.4f}, Bond Num RMSE: {train_b_num_rmse:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}, IF AUC: {val_if_auc:.4f}, Bond Type Acc: {val_bond_type_acc:.4f}, "
              f"Atom Type Acc: {val_atom_type_acc:.4f}, Atom Num RMSE: {val_a_num_rmse:.4f}, Bond Num RMSE: {val_b_num_rmse:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")

        epoch_model_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_model_path)

        plt.figure()
        plt.plot(range(1, epoch + 1), avg_train_losses, marker='o', label='Train Loss')
        plt.plot(range(1, epoch + 1), avg_val_losses, marker='o', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title(f'Average Loss vs Epoch (Up to Epoch {epoch})')
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"loss_vs_epoch_{epoch}.png"))
        plt.close()
 

if __name__ == "__main__":
    main()


 