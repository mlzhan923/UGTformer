import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
from scipy.sparse.linalg import eigs

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32) 
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)) 
    values = torch.from_numpy(sparse_mx.data) 
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def re_features(adj, features, K):    
    adj.requires_grad_(False)
    propagated_features = [features]

    for step in range(K):
        adj = adj.to_dense()  
        features = features.clone()  
        features = torch.matmul(adj, features)
        propagated_features.append(features)

    nodes_features = torch.stack(propagated_features, dim=1) 

    return nodes_features



