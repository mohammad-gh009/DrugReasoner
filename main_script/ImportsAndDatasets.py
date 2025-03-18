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
from sklearn.model_selection import train_test_split

# "/home/u111169/wrkdir/mgh-project/the_drugbank_daatset_to_replace_with_chemap_for_test.csv"

def train_valid_test_split(path2csv:str): 


    df = pd.read_csv(path2csv)

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
    dataset_train = Dataset.from_pandas(train_df)
    dataset_df = Dataset.from_pandas(val_df)
    dataset_test = Dataset.from_pandas(test_df)
    
    
    return train_df , val_df , test_df , dataset_train , dataset_df , dataset_test
    