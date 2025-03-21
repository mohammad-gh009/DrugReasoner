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
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

def evaluate(y_true ,y_pred ): 
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')#
    precision = precision_score(y_true, y_pred, average='weighted')#
    recall = recall_score(y_true, y_pred, average='weighted')#

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    # Calculate specificity (True Negative Rate)
    specificity = TN / (TN + FP)

    # Calculate NPV (Negative Predictive Value)
    NPV = TN / (TN + FN)

    auc = roc_auc_score(y_true, y_pred)

    results = {
        "F1 score": f1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
        "Specificity": specificity,
        # "NPV": NPV
    }
    return results

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir="/home/u111169/wrkdir/mgh-project/models")
num_added_toks = tokenizer.add_tokens(["<Approved>", "<NotApproved>"])
num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": ["<[lbl]>", "<[mr]>"]})
print("We have added", num_added_toks, "tokens")
model = AutoModelForCausalLM.from_pretrained("/home/u111169/wrkdir/mgh-project/models/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
model.resize_token_embeddings(len(tokenizer))
print("Imports completed")

os.environ["WANDB_DISABLED"] = "true"

df0 = pd.read_csv("/home/u111169/wrkdir/mgh-project/ChemAP/dataset/DrugApp/All_training_feature_vectors.csv")
df=df0[["SMILES",	"Label"]]
df = df[df['SMILES'].str.len() <= 750].reset_index(drop=True)

print("dataset created ")

from sklearn.model_selection import train_test_split
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


def convert_to_fingerprint(smiles, target):
    mol = Chem.MolFromSmiles(smiles)

    # Generate the Morgan fingerprint
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=256)
    morgan = ''.join(str(int(bit)) for bit in morgan_fp)
    if int(target) == 1:
        target = "<Approved>"
    else:
        target = "<NotApproved>"

    return smiles+ "<[mr]>" + morgan + "<[lbl]>" + str(target)+ tokenizer.eos_token


def convert_to_fingerprint_for_eval(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # Generate the Morgan fingerprint
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=256)
    morgan = ''.join(str(int(bit)) for bit in morgan_fp)

    return smiles+ "<[mr]>" + morgan + "<[lbl]>"

def convert_labels(target):

    if int(target) == 1:
        target = "<Approved>"
    else:
        target = "<NotApproved>"

    return tokenizer.encode(target)

# building the final dataset
train_df["SMILES"] = train_df.apply(lambda row: convert_to_fingerprint(row["SMILES"] , row["labels"]), axis=1)
# train_df["labels"] = train_df["labels"].map(convert_labels)
val_df["SMILES"] = val_df.apply(lambda row: convert_to_fingerprint_for_eval(row["SMILES"]), axis=1)
# val_df["labels"] = val_df["labels"].map(convert_labels)
test_df["SMILES"] = test_df.apply(lambda row: convert_to_fingerprint_for_eval(row["SMILES"]), axis=1)
# test_df["labels"] = test_df["labels"].map(convert_labels)

tokenizer.pad_token = tokenizer.eos_token
# #-----------------------------------
# # convert the dataframes to huggingface dataset for easier upload on hub and eaiser accessibility
dataset_train = Dataset.from_pandas(train_df[["SMILES"]])
dataset_df = Dataset.from_pandas(val_df[["SMILES"]])
dataset_test = Dataset.from_pandas(test_df[["SMILES"]])

def tokenize_function(examples):
    return tokenizer(examples['SMILES'], padding='max_length' , truncation=True )#,max_length=166

tokenized_train = dataset_train.map(tokenize_function)
tokenized_val = dataset_df.map(tokenize_function)
tokenized_test = dataset_test.map(tokenize_function)

import os
os.environ["WANDB_DISABLED"] = "true"

from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir='/home/u111169/wrkdir/mgh-project/checkpoints/druggpt2',
    num_train_epochs= 200,
    # load_best_model_at_end = True,
    # evaluation_strategy='steps',
    save_strategy='steps',
    learning_rate=2e-3,
    per_device_train_batch_size=8,
    # per_device_eval_batch_size=8,
    #eval_steps=20,
    #save_total_limit=3,
    gradient_accumulation_steps=4,
    # eval_accumulation_steps=1,
    # do_eval=True,
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
    # label_names=["approved" , "nonapproved"]
    # label_smoothing_factor=0.01,
)


from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    #eval_dataset=tokenized_val,
    #tokenizer = tokenizer,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
trainer.train(resume_from_checkpoint = True)