import pandas as pd
import numpy as np
from datasets import Dataset
import itertools
from datasets import load_dataset
from sentence_transformers import SentenceTransformer , SentenceTransformerTrainer
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss ,ContrastiveLoss, OnlineContrastiveLoss
import os
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import torch
from sklearn.model_selection import train_test_split
# Imports
import pandas as pd
from datasets import load_dataset , Dataset
from transformers import AutoModelForCausalLM , RobertaTokenizer , RobertaForSequenceClassification , TrainingArguments, Trainer,TrainerCallback,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support , accuracy_score
import torch
import numpy as np
import torch.nn as nn
import numpy as np
from datasets import Dataset, Features, Value,Sequence

print("Imports completed")

os.environ["WANDB_DISABLED"] = "true"

df0 = pd.read_csv("/home/u111169/wrkdir/mgh-project/ChemAP/dataset/DrugApp/All_training_feature_vectors.csv")
df=df0[["SMILES",	"Label"]]

print("dataset created ")

df.rename(columns={"Label":"labels"}, inplace=True)

test_size = 0.2
val_size = 0.5
train_df , temp = train_test_split(df , stratify = df.labels , test_size = test_size , random_state=1234)
test_df , val_df = train_test_split(temp , stratify = temp.labels , test_size = val_size , random_state=1234)
#-----------------------------------
# reset index
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


#-----------------------------------
# convert the dataframes to huggingface dataset for easier upload on hub and eaiser accessibility
dataset_train = Dataset.from_pandas(train_df)
dataset_df = Dataset.from_pandas(val_df)
dataset_test = Dataset.from_pandas(test_df)
#tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")#'FacebookAI/xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained("/home/u111169/.cache/huggingface/hub/models--liyuesen--druggpt/snapshots/9c03bc4c40f4c915fdf734396f3f437cc956c4b5")#'FacebookAI/xlm-roberta-large'


def tokenize_function(examples):
    return tokenizer(examples['SMILES'], padding="max_length" , truncation=True )#,max_length=166

tokenized_train = dataset_train.map(tokenize_function)
tokenized_val = dataset_df.map(tokenize_function)
tokenized_test = dataset_test.map(tokenize_function)

model_name ="DeepChem/ChemBERTa-77M-MTR"# "FacebookAI/xlm-roberta-large""google-bert/bert-base-uncased"
import os
os.environ["WANDB_DISABLED"] = "true"
model = AutoModelForCausalLM.from_pretrained("/home/u111169/.cache/huggingface/hub/models--liyuesen--druggpt/snapshots/9c03bc4c40f4c915fdf734396f3f437cc956c4b5")# ,num_labels=2)


from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir='/home/u111169/wrkdir/mgh-project/checkpoints/sft_chemberta',
    num_train_epochs= 100,
    load_best_model_at_end = True,
    evaluation_strategy='steps',
    save_strategy='steps',
    learning_rate=2e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=20,
    #save_total_limit=3,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    do_eval=True,
    do_train=True,
    weight_decay=0.1,
    logging_dir = "logs",
    logging_strategy="steps",
    logging_steps = 10,
    dataloader_drop_last=True,
    save_safetensors=False,
    adam_epsilon=1e-08,
    warmup_steps=100,
    seed=42,
    lr_scheduler_type='cosine',
    # label_smoothing_factor=0.01,
)


from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer = tokenizer,
)
trainer.train()