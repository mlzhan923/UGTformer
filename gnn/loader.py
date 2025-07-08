import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


# allowable node and edge features
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

def mol_to_graph_data(mol, graph_label):
    """
    Converts rdkit mol object to graph Data object for pytorch geometric,
    with masking and labeling for annotated atoms.
    
    :param mol: rdkit mol object
    :param normed_mapped_smiles: annotated SMILES with '1' as non-metabolic and '3' as metabolic sites
    :return: graph data object with attributes: x, edge_index, edge_attr, mask, and y (labels)
    """
    # atoms
    num_atom_features = 7  # atomic number, formal charge, chirality, hybridization, numH, implicit valence, degree
    atom_features_list = []
    mask = []
    labels = []

    for i, atom in enumerate(mol.GetAtoms()):
        atom_feature = [
            allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()),
            allowable_features['possible_formal_charge_list'].index(atom.GetFormalCharge()),
            allowable_features['possible_chirality_list'].index(atom.GetChiralTag()),
            allowable_features['possible_hybridization_list'].index(atom.GetHybridization()),
            allowable_features['possible_numH_list'].index(atom.GetTotalNumHs()),
            allowable_features['possible_implicit_valence_list'].index(atom.GetImplicitValence()),
            allowable_features['possible_degree_list'].index(atom.GetDegree())
        ]
        atom_features_list.append(atom_feature)
        
        # Check if atom is annotated in normed_mapped_smiles
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num == 1:  # Non-metabolic site
            mask.append(True)
            labels.append(0)
        elif atom_map_num == 3:  # Metabolic site
            mask.append(True)
            labels.append(1)
        else:  # Unannotated
            mask.append(False)
            labels.append(-1)  # -1 for unannotated atoms

    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.bool)
    y = torch.tensor(labels, dtype=torch.long)

    # bonds
    num_bond_features = 4
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features['possible_bonds'].index(bond.GetBondType()),
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
        graph_y = torch.tensor(graph_label, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)


    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mask=mask, y=y, graph_y=graph_y)

    return data



class MetabolicSiteDataset(InMemoryDataset):
    def __init__(self, root, smiles_list, labels, transform=None, pre_transform=None):
        self.smiles_list = smiles_list
        self.labels = labels
        super(MetabolicSiteDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.process()

    @property
    def processed_file_names(self):
        return 'processed_data.pt'

    def process(self):
        data_list = []
        for smiles, label in zip(self.smiles_list, self.labels):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                data = mol_to_graph_data(mol, label)
                data_list.append(data)
            else:
                print(f"Failed to convert molecule {smiles}, skipping.")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices
    
    @staticmethod
    def load_dataset(file_path):
        df = pd.read_csv(file_path)
        smiles_list = df['normed_mapped_smiles'].tolist()
        labels = df['label'].tolist()
        return smiles_list, labels




