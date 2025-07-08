import argparse
parser = argparse.ArgumentParser()
# Training settings
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=32,
                     help='input batch size for training (default: 32)')
parser.add_argument('--random_state', type=int, default=42, 
                    help='Random state for cross-validation')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                     help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0,
                    help='weight decay (default: 0)')
parser.add_argument('--num_layer', type=int, default=2,
                     help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=128,
                     help='embedding dimensions (default: 256)')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--JK', type=str, default="last",
                    help='how the node features are combined across layers. last, sum, max or concat')
parser.add_argument('--gnn_type', type=str, default="gin", help='type of GNN model (options: gin, gcn, gat, graphsage)')
parser.add_argument('--save_dir', type=str, default="baseline/saved_model/", help='Directory to save models')
args = parser.parse_args()

