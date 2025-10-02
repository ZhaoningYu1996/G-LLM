import os
from torch.utils.data import Dataset
import json
import csv
from rdkit import Chem
from tqdm import tqdm
import random
from collections import defaultdict
from utils.smiles2tree import smiles_to_tree, custom_json_serializer
from utils.utils import sanitize_smiles
from utils.sample_fragment import sample_fragment
# from torchtune.data import AlpacaInstructTemplate
from transformers import AutoTokenizer
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from utils.smiles2tu import to_tudataset
import pandas as pd

class G2TDataset(Dataset):
    def __init__(self, data_name, key, atom_format, bond_format, atom_type_enum, output_path):
        self.data_name = data_name
        ogb_dataset = PygGraphPropPredDataset(name=data_name)
        split_idx = ogb_dataset.get_idx_split()
        path = "ogb_datasets/raw/" + data_name + ".csv"
        # Read the CSV file
        raw_dataset = pd.read_csv(path)
        # Get the column names
        column_names = raw_dataset.columns.tolist()
        print(f"Column names: {column_names}")
        # Get the first 5 rows of the dataset
        print(f"First 5 rows: {raw_dataset.head()}")
        # print(stop)
        self.split_dataset(raw_dataset, split_idx, key)
        # self.prep_dataset(raw_data, key, valid_idx)
        self.format_data(atom_format, bond_format, atom_type_enum, output_path)

    def split_dataset(self, raw_data, split_idx, key):
        print("Splitting dataset...")
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        # Split the dataset into train, valid and test
        train_set = []
        valid_set = []
        test_set = []
        for idx, row in tqdm(raw_data.iterrows(), total=raw_data.shape[0]):
            smiles = row[key]
            smiles = sanitize_smiles(smiles)
            label = row['Class']
            if smiles is not None:
                if idx in train_idx:
                    train_set.append((smiles, label))
                elif idx in valid_idx:
                    valid_set.append((smiles, label))
                elif idx in test_idx:
                    test_set.append((smiles, label))

        print(f"Number of train data: {len(train_set)}")
        print(f"Number of valid data: {len(valid_set)}")
        print(f"Number of test data: {len(test_set)}")
        
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set



    def get_atom_set(self):
        atom_set = set([])
        for data in tqdm(self.dataset):
            mol = Chem.MolFromSmiles(data)
            for atom in mol.GetAtoms():
                atom_set.add(atom.GetSymbol())
        return atom_set
    
    def format_data(self, atom_format, bond_format, atom_type_enum, output_path: str):
        print("Sampling and formatting data...")
        # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
        tokenizer = AutoTokenizer.from_pretrained('unsloth/gemma-3-4b-it')
        # Define the schema for the Atom class
        train_data_list = []
        valid_data_list = []
        test_data_list = []

        input_set = set([])
        for smiles, label in tqdm(self.train_set):
            smiles = sanitize_smiles(smiles)
            if smiles is not None:
                json_data, analysis = smiles_to_tree(atom_format, bond_format, atom_type_enum, smiles)
                # print(json_data)
                # print(stop)
                input_data = json.dumps(json_data.dict(), indent=0)

                data = {
                    "instruction": f"Please analyze the structure and features of the molecule {smiles}, and predict the class of the molecule, the class should be 1 for the molecule binds to (i.e., inhibits) Beta-Secretase 1 or 0 for the molecule does not bind to (i.e., inhibits) Beta-Secretase 1.",
                    "input": f"The tree format of the molecule is as follows: {input_data}. We analyze the structure of the molecule: {analysis}. ",
                    "output": f"According to the analysis, the class of the molecule is {label}."
                }
                train_data_list.append(data)
                
                # data = {
                #     "instruction": f"Please analyze the structure and features of the molecule, and predict the class of the molecule, the class should be 1 for the molecule binds to (i.e., inhibits) Beta-Secretase 1 or 0 for the molecule does not bind to (i.e., inhibits) Beta-Secretase 1.",
                #     "input": f"The tree format of the molecule is as follows: {input_data}.",
                #     "output": f"{label}"
                # }
                # train_data_list.append(data)

                

        for smiles, label in tqdm(self.valid_set):
            smiles = sanitize_smiles(smiles)
            if smiles is not None:
                json_data, analysis = smiles_to_tree(atom_format, bond_format, atom_type_enum, smiles)

                input_data = json.dumps(json_data.dict(), indent=0)
                # data = {
                #     "instruction": f"Please predict the class of the molecule, it should be 1 for the molecule binds to (i.e., inhibits) Beta-Secretase 1 or 0 for the molecule does not bind to (i.e., inhibits) Beta-Secretase 1.",
                #     "input": f"The tree format of the molecule is as follows: {input_data}.",
                #     "output": f"{label}"
                # }
                data = {
                    "instruction": f"Please analyze the structure and features of the molecule {smiles}, and predict the class of the molecule, the class should be 1 for the molecule binds to (i.e., inhibits) Beta-Secretase 1 or 0 for the molecule does not bind to (i.e., inhibits) Beta-Secretase 1.",
                    "input": f"The tree format of the molecule is as follows: {input_data}. We analyze the structure of the molecule: {analysis}. ",
                    "output": f"According to the analysis, the class of the molecule is {label}."
                }
                valid_data_list.append(data)

        for smiles, label in tqdm(self.test_set):
            smiles = sanitize_smiles(smiles)
            if smiles is not None:
                json_data, analysis = smiles_to_tree(atom_format, bond_format, atom_type_enum, smiles)

                input_data = json.dumps(json_data.dict(), indent=0)
                # data = {
                #     "instruction": f"Please predict the class of the molecule, it should be 1 for the molecule binds to (i.e., inhibits) Beta-Secretase 1 or 0 for the molecule does not bind to (i.e., inhibits) Beta-Secretase 1.",
                #     "input": f"The tree format of the molecule is as follows: {input_data}.",
                #     "output": f"{label}"
                # }
                data = {
                    "instruction": f"Please analyze the structure and features of the molecule {smiles}, and predict the class of the molecule, the class should be 1 for the molecule binds to (i.e., inhibits) Beta-Secretase 1 or 0 for the molecule does not bind to (i.e., inhibits) Beta-Secretase 1.",
                    "input": f"The tree format of the molecule is as follows: {input_data}. We analyze the structure of the molecule: {analysis}. ",
                    "output": f"According to the analysis, the class of the molecule is {label}."
                }
                count = 0
                for key in data.keys():
                    if key in ['instruction', 'input', 'output']:
                        tokens = tokenizer.encode(data[key], add_special_tokens=True)  # Ensure special tokens are considered
                        count += len(tokens)
                if count < 26000:
                    test_data_list.append(data)
                else:
                    print(f"Text {smiles} exceeds 26000 tokens: {count}")

        # Save the train, valid and test data to JSON files
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'train.json'), 'w') as f:
            json.dump(train_data_list, f, indent=0)
        with open(os.path.join(output_path, 'valid.json'), 'w') as f:
            json.dump(valid_data_list, f, indent=0)
        with open(os.path.join(output_path, 'test.json'), 'w') as f:
            json.dump(test_data_list, f, indent=0)

# if __name__ == "__main__":
#     file_path = 'datasets/zinc250k_train.json'
#     raw_file_path = 'datasets/raw/zinc250k_property.csv'
#     download_raw_url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
#     valid_idx_path = 'datasets/raw/valid_idx_zinc250k.json'
#     download_index_url = None
#     dataset = ZINCDataset(file_path, raw_file_path, download_raw_url, valid_idx_path, download_index_url)