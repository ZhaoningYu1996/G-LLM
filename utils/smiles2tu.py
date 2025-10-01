from utils.utils import get_mol
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolops
from utils.mapping_conf import ATOM, EDGE
from tqdm import tqdm

def to_tudataset(smiles, data_name, label=None):
    if data_name == "Mutagenicity":
        mol = get_mol(smiles, True)
    else:
        mol = get_mol(smiles, False)
    if mol == None:
        return None
    if mol.GetNumAtoms() == 0 and mol.GetNumBonds() == 0:
        return None
    rdmolops.AssignStereochemistry(mol)
    # if addH == True:
    #     mol = Chem.AddHs(mol)
    # Extract atom-level features
    atom_features = []
    swapped_feature_map = {value: key for key, value in ATOM[data_name].items()}
    for atom in mol.GetAtoms():
        atom_features.append(swapped_feature_map[atom.GetAtomicNum()])

    # Extract bond-level features
    bond_features = []
    swapped_edge_feature_map = {value: key for key, value in EDGE[data_name].items()}
    for bond in mol.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()
        if bond_type == 1.0:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.SINGLE]
        elif bond_type == 1.5:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.AROMATIC]
        elif bond_type == 2.0:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.DOUBLE]
        elif bond_type == 3.0:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.TRIPLE]
        else:
            bond_type = swapped_edge_feature_map[None]

        bond_features.append(bond_feat)
        bond_features.append(bond_feat)
    
    atom_features = torch.tensor(atom_features, dtype=torch.long)
    x = F.one_hot(atom_features, num_classes=len(swapped_feature_map)).float()  # Node feature matrix
    edge_index = []
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Edge connectivity
    if mol.GetNumBonds() == 0:
        edge_index = torch.tensor([[], []], dtype=torch.long)
    # if mol.GetNumBonds() == 0:
        # edge_index.fill_([])
    bond_features = torch.tensor(bond_features, dtype=torch.long)  # Edge feature matrixd
    edge_attr = F.one_hot(bond_features, num_classes=len(swapped_edge_feature_map))
    if not label == None:
        y = torch.tensor([label])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data