import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset , Dataset
import itertools
from datasets import load_dataset
from sentence_transformers import SentenceTransformer , SentenceTransformerTrainer
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss ,ContrastiveLoss, OnlineContrastiveLoss
import os
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import torch
from transformers import RobertaTokenizer , RobertaForSequenceClassification , TrainingArguments , Trainer,TrainerCallback,AutoModelForSequenceClassification,AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import pandas as pd
import torch
from transformers import AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

df0 = pd.read_csv("/home/u111169/wrkdir/mgh-project/ChemAP/dataset/DrugApp/All_training_feature_vectors.csv")
df=df0[["SMILES",	"Label"]]

print("dataset created ")

from sklearn.model_selection import train_test_split
test_size = 0.2
# val_size = 0.5
train_df , test_df = train_test_split(df , stratify = df.Label , test_size = test_size , random_state=1234)

print("train test done!")

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