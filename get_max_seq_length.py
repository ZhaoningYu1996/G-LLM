import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json

argparser = argparse.ArgumentParser(description='Get the maximum sequence length')
argparser.add_argument('--data_path', type=str, default='test_data/BACE/test.json', help='Path to save the processed data')
args = argparser.parse_args()

json_list = json.load(open(args.data_path, 'r'))
print(len(json_list))

# Example for loading a dataset
dataset = load_dataset('json', data_files=args.data_path)
print(dataset)
texts = dataset['train']  # Adjust according to your dataset structure

print(f"Number of texts: {len(texts)}")

# tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('unsloth/gemma-3-4b-it')

max_seq_len = 0

seq_len_list = []

# Collect all text that has length less than 7500
cleaned_texts = []

# Calculate max sequence length
counting = 0
for i, text in enumerate(tqdm(texts, desc="Calculating max sequence length", total=len(texts))):
    # print(text)
    # print(text.keys())
    # print(stop)
    count = 0
    for key in text.keys():
        if key in ['instruction', 'input', 'output']:
        # if key == 'output':
            # max_input_len = 0
            # if key == 'input':
            #     for i in range(len(text[key])):
            #         tokens = tokenizer.encode(text[key][i], add_special_tokens=True)
            #         max_input_len = max(max_input_len, len(tokens))
            #     count += max_input_len
            # else:
            tokens = tokenizer.encode(text[key], add_special_tokens=True)  # Ensure special tokens are considered
            count += len(tokens)
        # else:
        #     print(f"Key: {key}")
        #     print(stop)
    max_seq_len = max(max_seq_len, count)
    seq_len_list.append(count)
    if count > 24000:
        print(f"Text {i} exceeds 25000 tokens: {count}")
        counting += 1
        # print(text)
        # print(stop)
    # if count <= 7500:
    #     cleaned_texts.append(text)

print(f"Maximum sequence length: {max_seq_len}")
print(f"Number of texts with length greater than 25000: {counting}")