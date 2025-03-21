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
print("Imports completed")

os.environ["WANDB_DISABLED"] = "true"

df0 = pd.read_csv("/home/u111169/wrkdir/mgh-project/ChemAP/dataset/DrugApp/All_training_feature_vectors.csv")
df=df0[["SMILES",	"Label"]]

print("dataset created ")

from sklearn.model_selection import train_test_split
test_size = 0.2
# val_size = 0.5
train_df , test_df = train_test_split(df , stratify = df.Label , test_size = test_size , random_state=1234)

print("train test done!")
# test_df , val_df = train_test_split(temp , stratify = temp.Label , test_size = val_size , random_state=1234)
#-----------------------------------
# reset index
train_df.reset_index(drop=True, inplace=True)
# val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

def duable_maker_with_label(df):
    pos_df = df[df["Label"]==0]["SMILES"]
    neg_df = df[df["Label"]==1]["SMILES"]
    elements = pos_df.tolist()
    elements_neg = neg_df.tolist()
    tuple_length = 2
    unique_tuples = list(itertools.combinations(elements, tuple_length))
    unique_tuples_neg = list(itertools.combinations(elements_neg, tuple_length))
    ll = unique_tuples + unique_tuples_neg
    df1 = pd.DataFrame(unique_tuples, columns=['Column1', 'Column2'])
    df2 = pd.DataFrame(unique_tuples, columns=['Column1', 'Column2'])
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df["label"]=1
    unique_tuples = set(itertools.product(elements, elements_neg))
    unique_tuples_list = list(unique_tuples)
    df3 = pd.DataFrame(unique_tuples_list, columns=['Column1', 'Column2'])
    df3["label"]=0
    combined_df_f = pd.concat([combined_df, df3], ignore_index=True)
    return combined_df_f

df_embed_d = duable_maker_with_label(train_df)
train_df_d , val_df_d = train_test_split(df_embed_d , stratify = df_embed_d.label , test_size = 0.2 , random_state=1234)
train_df_d.reset_index(drop=True, inplace=True)
val_df_d.reset_index(drop=True, inplace=True)
train_dataset = Dataset.from_pandas(train_df_d)
val_dataset = Dataset.from_pandas(val_df_d)

print("datasets are doubled and are ready for training")

# Load a model to train/finetune
model = SentenceTransformer("/home/u111169/wrkdir/mgh-project/models/models--DeepChem--ChemBERTa-77m-MTR/snapshots/66b895cab8adebea0cb59a8effa66b2020f204ca") # "Muennighoff/SGPT-125M-weightedmean-nli-bitfit" "microsoft/mpnet-base"
print("Max Sequence Length:", model.max_seq_length)

loss = OnlineContrastiveLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir='/home/u111169/wrkdir/mgh-project/checkpoints',
    seed = 42 ,
    data_seed = 42,
    do_eval=True,
    do_train=True,
    num_train_epochs= 10,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=128,
    gradient_checkpointing=True ,
    eval_accumulation_steps= 1,
    eval_strategy = "steps",
    #drop_last=False,
    eval_steps=400,
    eval_on_start= False,
    weight_decay=0.01,
    learning_rate=2e-3,
    lr_scheduler_type='cosine',
    warmup_steps=100,
    logging_strategy = "steps",
    logging_steps= 200 ,
    save_strategy='steps',
    save_total_limit = 4,
    save_safetensors = False ,
    #torch_empty_cache_steps= 60,
    torchdynamo= "eager" ,
    torch_compile= True,
)
trainer = SentenceTransformerTrainer(
    model=model,
    args = args ,
    train_dataset=train_dataset,
    eval_dataset= val_dataset,
    loss = loss
)
trainer.train()
