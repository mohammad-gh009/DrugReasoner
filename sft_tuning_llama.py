# %%capture
# ! pip install trl
# ! pip install datasets
# ! pip install rdkit
# ! pip install peft
# ! pip install git+https://github.com/bitsandbytes-foundation/bitsandbytes.git



from datasets import load_dataset , Dataset
from trl import SFTConfig, SFTTrainer
import pandas as pd
df_sft_sft = pd.read_csv("/content/drive/MyDrive/Papers/Original_Papers/Llm_drug_prediction/Code/reasoning_data_api/just_for_sft.csv")

df_sft_sft_100 = pd.concat(100*[df_sft_sft], axis=0, ignore_index=True)

dset = Dataset.from_pandas(df_sft_sft_100)

def get_tokenize(dset):
    return tokenizer(dset["sft_input"] , padding="max_length" , truncation=True  )


dset2train = dset.map(get_tokenize)
def get_sft_prompt(input , thinking , answer):
    return f"""<｜begin▁of▁sentence｜><｜User｜>{input}<think>{thinking}</think><｜Assistant｜>{answer}<｜end▁of▁sentence｜><｜Assistant｜>"""


df_sft["sft_input"] = df_sft.apply(lambda row : get_sft_prompt(row["1st_prompt"] ,f"""{row["think1"]} \n{row["answer1"]} \n but wait lets analysis again \n{row["think2"]}\n {row["answer2"]}"""  , row["anwer3"]) , axis = 1)
# Load model directly
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,TrainingArguments

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Actually 4-bit, but commonly referred to as "8-bit" quantization
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", # this is not working I shoud replace it with 7b
                                             quantization_config=bnb_config,
                                             device_map="auto",  # Automatically places layers on available devices
                                             torch_dtype=torch.bfloat16,
                                             ) #"manycore-research/SpatialLM-Llama-1B""deepseek-ai/DeepSeek-R1"

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,  # Rank of the low-rank matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj" , "up_proj" , "gate_proj"],  # Modules to apply LoRA to
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 6. Wrap model with LoRA
model = get_peft_model(model, peft_config)

[len(tokenizer.tokenize(i)) for i in df_sft["sft_input"]]
df_sft_sft = df_sft[["sft_input"]]
df_sft_sft.to_csv("/content/drive/MyDrive/Papers/Original_Papers/Llm_drug_prediction/Code/reasoning_data_api/just_for_sft.csv", index = False)
training_args = SFTConfig(
    output_dir="/tmp",                  # Directory for output files
    num_train_epochs=5,                # Number of training epochs
    per_device_train_batch_size=2,     # Batch size per device
    gradient_accumulation_steps=16,     # Number of updates steps to accumulate
    learning_rate=2e-4,                # Learning rate
    weight_decay=0.01,                 # Weight decay
    logging_dir="./logs",              # Directory for logs
    logging_steps=1,                  # Log every X steps
    save_steps=500,                    # Save checkpoint every X steps
    save_total_limit=2,                # Max number of checkpoints to keep
    # evaluation_strategy="steps",       # Evaluate every X steps
    # eval_steps=100,                    # Evaluation steps
    fp16=True,                         # Use mixed precision training
    warmup_steps=500,                  # Number of warmup steps
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    report_to = "none"
)

trainer = SFTTrainer(
    model = model ,
    train_dataset=dset2train,
    args=training_args,
)
trainer.train()