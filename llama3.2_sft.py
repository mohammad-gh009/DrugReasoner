# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir="/home/u111169/wrkdir/mgh-project/models",trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir="/home/u111169/wrkdir/mgh-project/models", trust_remote_code=True)

# Imports
import pandas as pd
from datasets import load_dataset , Dataset
from transformers import  TrainingArguments , Trainer,TrainerCallback,AutoModelForSequenceClassification,AutoTokenizer
from huggingface_hub import HfApi, create_repo
from sklearn.metrics import precision_recall_fscore_support , accuracy_score
import torch
import numpy as np
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from datasets import Dataset, Features, Value,Sequence
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from trl import SFTConfig , SFTTrainer


num_added_toks = tokenizer.add_tokens(["<Approved>", "<NotApproved>"])
num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": ["<[lbl]>", "<[mr]>"]})
print("We have added", num_added_toks, "tokens")
model.resize_token_embeddings(len(tokenizer))


from rdkit import Chem

def get_canonical_smiles(smiles):
    # Convert input SMILES to molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None  # Invalid SMILES
    
    # Generate canonical SMILES
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return canonical_smiles


df_git = pd.read_csv("/home/u111169/wrkdir/mgh-project/ChemAP/dataset/DrugApp/All_training_feature_vectors.csv")
df = df_git[["SMILES","Label"]]
#df.rename(columns={"Label":"labels"}, inplace=True)
df["SMILES"] = df["SMILES"].apply(lambda x: get_canonical_smiles(x))


#-----------------------------------
#split dataset into train , test and val(just useing in the training process)
from sklearn.model_selection import train_test_split
test_size = 0.2
val_size = 0.5
train_df , temp = train_test_split(df , stratify = df.Label , test_size = test_size , random_state=1234)


test_df , val_df = train_test_split(temp , stratify = temp.Label , test_size = val_size , random_state=1234)
#-----------------------------------
# reset index
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


def convert_to_fingerprint(smiles, target):
    mol = Chem.MolFromSmiles(smiles)

    # Generate the Morgan fingerprint
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=2048)
    morgan = ''.join(str(int(bit)) for bit in morgan_fp)
    if int(target) == 1:
        target = "<Approved>"
    else:
        target = "<NotApproved>"

    return smiles+ "<[mr]>" + morgan + "<[lbl]>" + str(target)+ tokenizer.eos_token


def convert_to_fingerprint_for_eval(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # Generate the Morgan fingerprint
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=2048)
    morgan = ''.join(str(int(bit)) for bit in morgan_fp)

    return smiles+ "<[mr]>" + morgan + "<[lbl]>"

# building the final dataset
train_df["text"] = train_df.apply(lambda row: convert_to_fingerprint(row["SMILES"], row["Label"]), axis=1)
val_df["text"] = val_df.apply(lambda row: convert_to_fingerprint_for_eval(row["SMILES"]), axis=1)
test_df["text"] = test_df.apply(lambda row: convert_to_fingerprint_for_eval(row["SMILES"]), axis=1)


dataset_train = Dataset.from_pandas(train_df[["text"]])
dataset_val = Dataset.from_pandas(val_df[["text"]])
dataset_test = Dataset.from_pandas(test_df[["text"]])

tokenizer.pad_token = tokenizer.eos_token 

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True )#,max_length=166


tokenized_train = dataset_train.map(tokenize_function)
tokenized_val = dataset_val.map(tokenize_function)
tokenized_test = dataset_test.map(tokenize_function)


args = SFTConfig(
    output_dir='/home/u111169/wrkdir/mgh-project/checkpoints/llama3.2_sft',
    seed = 42 ,
    data_seed = 42,
    do_eval=True,
    do_train=True,
    #Sizes
    num_train_epochs= 5,
    per_device_train_batch_size=4,
    # per_device_eval_batch_size=2,
    # evaluations  :
    # eval_strategy = "steps",
    # eval_steps=20,
    # eval_on_start= False,
    #Gradient
    prediction_loss_only = False,
    gradient_accumulation_steps=4,
    weight_decay=0.1,
    # Learning rare
    learning_rate=2e-3,
    lr_scheduler_type='cosine',
    warmup_steps=50,
    # Logging and Saving:
    logging_dir = "/home/u111169/wrkdir/mgh-project/checkpoints/llama3.2_sft/log",
    logging_strategy = 'steps',
    # logging_first_step = ,
    logging_steps= 5 ,
    save_strategy='steps',
    save_steps = 20,
    #save_total_limit = ,
    save_safetensors = False ,
    # device :
    torch_empty_cache_steps= 4 ,
    # Torch compile
    torch_compile= True,
    max_seq_length=1700,
)


trainer = SFTTrainer(
    model = model,
    train_dataset=tokenized_train,
    # eval_dataset = dataset_val ,
    args=args,
)


trainer.train()