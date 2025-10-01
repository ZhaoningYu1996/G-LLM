import argparse
import pandas as pd
import json
from ogb_datasets.ogb_dataset import G2TDataset
from utils.format import create_mol_format
from utils.config import AtomTypeEnumZinc, BondTypeEnumZinc, AtomTypeEnumQM9, BondTypeEnumQM9, AtomTypeEnum, BondTypeEnum
from torch_geometric.datasets import TUDataset
from utils.pyg2smiles import to_smiles

argparser = argparse.ArgumentParser(description='Preprocess the data')
argparser.add_argument('--data_name', type=str, default='ogbg-molbace', help='Name of the dataset')
argparser.add_argument('--output_path', type=str, default='data/processed/ogb/', help='Path to save the processed data')
args = argparser.parse_args()

if args.data_name == 'ogbg-molbace':
    atom_type_enum = AtomTypeEnum
    bond_type_enum = BondTypeEnum
    atom_format, bond_format = create_mol_format(atom_type_enum, bond_type_enum)
    key = 'smiles'
    
elif args.data_name == 'hiv':
    raw_file_path = 'data/raw/rand_train_smiles.csv'
    raw_data = pd.read_csv(raw_file_path)
    valid_idx = []
    atom_type_enum = AtomTypeEnum
    bond_type_enum = BondTypeEnum
    atom_format, bond_format = create_mol_format(atom_type_enum, bond_type_enum)
    key = 'smiles'
elif args.data_name == 'Mutagenicity':
    dataset = TUDataset(root='datasets/', name='Mutagenicity')
    smiles_list = []
    label_list = []
    for data in dataset:
        smiles = to_smiles(data, kekulize=True, data_name='Mutagenicity')
        if smiles is None:
            continue
        smiles_list.append(smiles)
        label_list.append(data.y.item())
    # Save it to a CSV file
    raw_data = pd.DataFrame({'smiles': smiles_list, 'label': label_list})
    print(len(smiles_list))
    print(raw_data.head())

    valid_idx = []
    atom_type_enum = AtomTypeEnum
    bond_type_enum = BondTypeEnum
    atom_format, bond_format = create_mol_format(atom_type_enum, bond_type_enum)
    key = 'smiles'
else:
    raise ValueError(f"Invalid dataset name: {args.data_name}")

output_path = args.output_path + args.data_name + '/'

G2TDataset(data_name=args.data_name, key=key, atom_format=atom_format, bond_format=bond_format, atom_type_enum=atom_type_enum, output_path=output_path)