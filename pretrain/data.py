import utils
import dgl
import torch
import scipy.sparse as sp
import numpy as np
from torch_geometric.data import InMemoryDataset
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import Dataset
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import AllChem
import pandas as pd
from rdkit import Chem

# 定义原子特征集合
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
        'misc'
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 'misc'
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        'misc'
    ],
    'possible_bond_stereo_list':[
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        ],
    'possible_is_conjugated_list': [False, True],
    'possible_is_in_ring_list': [False, True] 

}

def safe_index(l, e):
    try:
        return l.index(e)
    except:
        return len(l) - 1
    
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol

def mol_to_graph_data(mol):
    
    atom_features_list = []

    for i, atom in enumerate(mol.GetAtoms()):
        atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_chirality_list'] , str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_degree_list'], atom.GetDegree()), 
        ]
        atom_features_list.append(atom_feature)
    
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    edges_list = []
    edge_features_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_feature = [
                safe_index(allowable_features['possible_bonds'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(bond.GetStereo()),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                allowable_features['possible_is_in_ring_list'].index(bond.IsInRing())
            ]
        
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)

    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        full_atom_feature_dims = [
            119,  # num_atom_type
            12,   # num_formal_charge
            5,    # num_chirality_tag
            7,    # num_hybridization
            10,    # num_numH
            7,    # num_implicit_valence
            12,   # num_degree
        ]

        self.atom_embedding_list = torch.nn.ModuleList()

        for dim in full_atom_feature_dims:
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)


    def forward(self, x):
        x_embedding = 0
        for i in range(len(self.atom_embedding_list)):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding 
    
class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        full_bond_feature_dims = [
            5,
            6,
            2,
            2
        ]

        self.bond_embedding_list = torch.nn.ModuleList()

        for dim in full_bond_feature_dims:
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding

    
class MolGraph(object):
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        graph_data = mol_to_graph_data(self.mol)
        self.x = graph_data.x
        self.edge_index = graph_data.edge_index
        self.edge_attr = graph_data.edge_attr

        self.num_part = self.size_atom()

    def size_atom(self): 
        return self.x.size()[0]

    def size_bond(self): 
        return self.edge_attr.size()[0]
    

class MoleculeDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        smiles = self.data[idx] 

        mol_graph = MolGraph(smiles) 

        return mol_graph
    
def molgraph_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        data = Data(x=mol.x, edge_index=mol.edge_index, edge_attr=mol.edge_attr, num_part=mol.num_part)
        graph_data_batch.append(data)
    new_batch = Batch().from_data_list(graph_data_batch) 
    return new_batch
    





