from unsloth import FastLanguageModel
import json
import os
from tqdm import tqdm
from transformers import TextStreamer
import random
from datasets import load_dataset
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name = "saved_models/gemma-3-4b-it_sft_12_16B_BACE_v_2",
    max_seq_length = 32000,
    load_in_4bit = False,
    load_in_8bit = False,
)

FastModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# input_path = 'test_data/BACE/test.json'
# input_path = 'valid_data/BACE/valid.json'
input_path = 'sft_data/BACE/train.json'

dataset = load_dataset("json", data_files=input_path, split="train")
print(f"Length of dataset: {len(dataset)}")

preds = []
labels = []

out_path = "predictions/BACE/"
if not os.path.exists(out_path):
    os.makedirs(out_path)
for i in tqdm(range(len(dataset))):
    labels.append(dataset[i]['output'])
    input = tokenizer([alpaca_prompt.format(dataset[i]['instruction'], dataset[i]['input'], "")], return_tensors = "pt").to("cuda")
    # text_streamer = TextStreamer(tokenizer)
    generated_ids = model.generate(**input, max_new_tokens = 64, do_sample=False, temperature = 1.0, top_p = 1, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    prompt = tokenizer.decode(input["input_ids"][0], skip_special_tokens=True)
    preds.append(output[0][len(prompt):])

# Save the output and label to a csv file using pandas
import pandas as pd
df = pd.DataFrame({"predictions": preds, "labels": labels})
df.to_csv(out_path + "sft_12_16B_v_2_train.csv", index=False)