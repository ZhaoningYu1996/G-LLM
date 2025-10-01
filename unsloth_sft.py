from unsloth import FastLanguageModel, FastModel

import torch
max_seq_length = 32000 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastModel.from_pretrained(
    # model_name = "Qwen/Qwen2.5-3B-Instruct",
    # model_name = "unsloth/gemma-3-4b-it",
    model_name = "saved_models/gemma-3-4b-it_sft_15_16B_BACE_v_2",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    load_in_8bit = False,
    full_finetuning= False, # Set to True for full finetuning
)

# model = FastLanguageModel.get_peft_model(
#     model,
#     r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = [
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ], # Remove QKVO if out of memory
#     lora_alpha = lora_rank,
#     use_gradient_checkpointing = "unsloth", # Enable long context finetuning
#     random_state = 3407,
# )

# model = FastModel.get_peft_model(
#     model,
#     finetune_vision_layers     = False, # Turn off for just text!
#     finetune_language_layers   = True,  # Should leave on!
#     finetune_attention_modules = True,  # Attention good for GRPO
#     finetune_mlp_modules       = True,  # SHould leave on always!

#     r = 32,           # Larger = higher accuracy, but might overfit
#     lora_alpha = 32,  # Recommended alpha == r at least
#     lora_dropout = 0,
#     bias = "none",
#     random_state = 3407,
# )

# Data Prep
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

from datasets import load_dataset
dataset = load_dataset("json", data_files="sft_data/BACE/train.json", split="train")
print(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)

# Train the model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# response_template = "### Response:\n"
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 3,
    packing = False, # Can make training 5x faster for short sequences.
    # data_collator=collator,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2,
        warmup_steps = 50,
        num_train_epochs = 3, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 2e-4,
        
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "### Instruction:",
    response_part = "### Response:",
)
trainer_stats = trainer.train()

model.save_pretrained("saved_models/gemma-3-4b-it_sft_18_16B_BACE_v_2")  # Local saving
tokenizer.save_pretrained("saved_models/gemma-3-4b-it_sft_18_16B_BACE_v_2")  # Local saving
# model.save_pretrained_merged("saved_models/Qwen2_5_3B_Instruct_sft_10_16B_Mutagenicity", tokenizer, save_method = "lora",)