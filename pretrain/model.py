import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.utils import to_dense_batch
from data import AtomEncoder, BondEncoder
import utils
import dgl
import torch
import scipy.sparse as sp

def get_attn_pad_mask(mask):
    batch_size, seq_len = mask.size()
    mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
    pad_attn_mask = mask * mask.transpose(1, 2)
    return pad_attn_mask.eq(0)  


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_() 
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

class CentralityEncoding(nn.Module):
    def __init__(self, out_dim):
        super(CentralityEncoding, self).__init__()
        self.linear = nn.Linear(1, out_dim)

    def forward(self, dgl_graph):
        deg = dgl_graph.in_degrees().float().unsqueeze(1)
        deg = deg.to(next(self.linear.parameters()).device)
        centrality_enc = self.linear(deg)  # (N, out_dim)
        return centrality_enc

class TransformerModel(nn.Module):
    def __init__(
        self,
        hops, 
        n_class,
        input_dim, 
        pe_dim,
        emb_dim,
        n_layers=6,
        num_heads=8,
        hidden_dim=64,
        ffn_dim=64, 
        dropout_rate=0,
        attention_dropout_rate=0.1
    ):
        super().__init__()

        self.seq_len = hops + 1 
        self.pe_dim = pe_dim
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        
        self.n_layers = n_layers
        self.n_class = n_class

        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.alpha_param = nn.Parameter(torch.tensor(0.5))

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim) 

       
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        self.centrality_encoder = CentralityEncoding(pe_dim)
        
        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads, edge_feature_size=self.emb_dim)
                    for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders) 
        self.final_ln = nn.LayerNorm(hidden_dim)

        
        self.node_encoder_layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=self.hidden_dim,
                ffn_size=self.ffn_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                num_heads=self.num_heads
            )
            for _ in range(self.n_layers)
        ])


        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1) 
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        all_features_list = [] 
        all_bond_features_list = []

        num_graphs = batched_data.batch.max().item() + 1

        device = batched_data.x.device


        for i in range(num_graphs):  
            node_mask = batched_data.batch == i
            node_features = batched_data.x[node_mask]  

            embedded_features = self.atom_encoder(node_features) 

            edge_mask = batched_data.batch[batched_data.edge_index[0]] == i  
            edge_index = batched_data.edge_index[:, edge_mask] 

            edge_index = edge_index.cpu() 

            edge_attr = batched_data.edge_attr[edge_mask] 
            node_indices = torch.where(node_mask)[0]
            global_to_local = {idx.item(): j for j, idx in enumerate(node_indices)}

            edge_index = torch.tensor([[global_to_local[src.item()], global_to_local[dst.item()]] 
                               for src, dst in edge_index.T if src.item() in global_to_local and dst.item() in global_to_local]).T
            
            bond_features = self.bond_encoder(edge_attr)

            all_bond_features_list.append(bond_features)

            adj = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                        shape=(node_features.size(0), node_features.size(0)))
            adj = adj + sp.eye(adj.shape[0]) 

            D1 = np.array(adj.sum(axis=1)) ** (-0.5)
            D2 = np.array(adj.sum(axis=0)) ** (-0.5)
            D1 = sp.diags(D1[:, 0], format='csr')
            D2 = sp.diags(D2[0, :], format='csr')
            adj_normalized = D1 @ adj @ D2

            graph = dgl.from_scipy(adj_normalized)
            graph = dgl.to_bidirected(graph)

            centrality_enc = self.centrality_encoder(graph)
            features_with_centrality = torch.cat([embedded_features, centrality_enc], dim=1)
            
            adj_normalized = utils.sparse_mx_to_torch_sparse_tensor(adj_normalized).to(embedded_features.device)

            propagated_features = utils.re_features(adj_normalized, features_with_centrality, self.seq_len - 1)

            all_features_list.append(propagated_features)
   

        
        all_features = torch.cat(all_features_list, dim=0)
        all_bond_features = torch.cat(all_bond_features_list, dim=0)
       

        tensor = self.att_embeddings_nope(all_features) 
        edge_index = batched_data.edge_index
        batch = batched_data.batch
        for enc_layer in self.layers:
            tensor = enc_layer(tensor, edge_attr=all_bond_features, edge_index = edge_index, batch= batch, attn_mask=None)
           
        output = self.final_ln(tensor) 
   
        target = output[:,0,:].unsqueeze(1).repeat(1,self.seq_len-1,1) 
        

        split_tensor = torch.split(output, [1, self.seq_len-1], dim=1) 
        node_tensor = split_tensor[0] 
        neighbor_tensor = split_tensor[1]

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1) 

        neighbor_tensor = neighbor_tensor * layer_atten 
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True) 
        output = (node_tensor + neighbor_tensor).squeeze() 
       
        x_batch, mask = to_dense_batch(output, batched_data.batch)
        attn_mask = get_attn_pad_mask(mask)

        for node_enc_layer in self.node_encoder_layers:
            x_batch = node_enc_layer(x_batch, edge_attr=None, edge_index=None, batch=None, attn_mask=attn_mask)

        x_batch = x_batch[mask] 
        node_embeddings = x_batch
        
        graph_embeddings = global_max_pool(node_embeddings, batch)

        return node_embeddings, graph_embeddings
        

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size) 
        self.gelu = nn.GELU() 
        self.layer2 = nn.Linear(ffn_size, hidden_size) 

    def forward(self, x):
        x = self.layer1(x) 
        x = self.gelu(x)
        x = self.layer2(x) 
        return x 


def to_local_edge_index(edge_index, batch):
    node_offset = torch.cumsum(torch.bincount(batch), dim=0)[:-1]
    node_offset = torch.cat([torch.tensor([0], device=edge_index.device), node_offset])

    local_edge_index = edge_index.clone()
    for i, offset in enumerate(node_offset):
        mask = (batch[edge_index[0]] == i)
        local_edge_index[:, mask] -= offset

    return local_edge_index, batch[edge_index[0]]


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, edge_feature_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        if edge_feature_size > 0: 
            self.edge_proj = nn.Linear(edge_feature_size, num_heads)  
        else:
            self.edge_proj = None 
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, edge_attr=None, edge_index=None, batch=None, attn_mask=None):
        orig_q_size = q.size()  
        batch_size, seq_len, _ = q.size()

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.att_size).transpose(1, 2)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.att_size).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.att_size).transpose(1, 2)

        if edge_attr is not None and self.edge_proj is not None:
            edge_weights = self.edge_proj(edge_attr)  
            src_nodes, dst_nodes = edge_index
            graph_id = batch[edge_index[0]]  

            edge_bias = torch.zeros((batch_size, self.num_heads, seq_len, seq_len), device=edge_attr.device)
            for head in range(self.num_heads):
                sparse_edge_bias = torch.sparse_coo_tensor(
                    indices=torch.stack([graph_id, src_nodes, dst_nodes]),
                    values=edge_weights[:, head],
                    size=(batch_size, seq_len, seq_len),
                    device=edge_attr.device,
                )
                edge_bias[:, head, :, :] = sparse_edge_bias.to_dense()

            edge_bias = edge_bias.permute(0, 1, 2, 3)  
        else:
            edge_bias = 0  

        q = q * self.scale
        k_t = k.transpose(2, 3)  
        attn_scores = torch.matmul(q, k_t)  

        attn_scores = attn_scores + edge_bias

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  
            attn_scores.masked_fill_(attn_mask, -1e9)

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.att_dropout(attn)
        out = torch.matmul(attn, v)  

        out = out.transpose(1, 2).contiguous()  
        out = out.view(batch_size, -1, self.num_heads * self.att_size)  
        out = self.output_layer(out)

        assert out.size() == orig_q_size
        return out

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, edge_feature_size=None):
        super(EncoderLayer, self).__init__()
        self.use_edge_attr = edge_feature_size is not None  

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size,
            edge_feature_size if self.use_edge_attr else 0,  
            attention_dropout_rate,
            num_heads,
        )
        self.self_attention_dropout = nn.Dropout(attention_dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_attr=None, edge_index=None, batch=None, attn_mask=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, edge_attr=edge_attr, edge_index=edge_index, batch=batch, attn_mask=attn_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x







