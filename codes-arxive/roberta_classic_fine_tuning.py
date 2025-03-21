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
from transformers import RobertaTokenizer , RobertaForSequenceClassification , TrainingArguments , Trainer,TrainerCallback,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support , accuracy_score
import torch
import numpy as np
import torch.nn as nn
import numpy as np
from datasets import Dataset, Features, Value,Sequence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score


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
tokenizer = RobertaTokenizer.from_pretrained("/home/u111169/.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots/66b895cab8adebea0cb59a8effa66b2020f204ca")#'FacebookAI/xlm-roberta-large'


def tokenize_function(examples):
    return tokenizer(examples['SMILES'], padding="max_length" , truncation=True )#,max_length=166

tokenized_train = dataset_train.map(tokenize_function)
tokenized_val = dataset_df.map(tokenize_function)
tokenized_test = dataset_test.map(tokenize_function)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor([len(train_df[train_df["labels"] == i]) / len(train_df) for i in np.unique(train_df["labels"])]).to(device)
#class_weights = torch.tensor([0.25,0.25,0.25,0.25]).to(device)

class CustomLoss(nn.Module):
    def __init__(self, class_weights):
        super(CustomLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        l1_loss = torch.mean(torch.abs(logits))
        return ce_loss + 0.01 * l1_loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = CustomLoss(class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
    


model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-base", cache_dir="/home/u111169/wrkdir/mgh-project/models" ,num_labels=2)
model.config.classifier_dropout=0.01

os.environ["WANDB_DISABLED"] = "true"



from transformers import EarlyStoppingCallback
training_args = TrainingArguments(
    output_dir='/home/u111169/wrkdir/mgh-project/checkpoints/roberta_classic',
    num_train_epochs= 10,
    evaluation_strategy='steps',
    save_strategy='steps',
    learning_rate=2e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_steps=20,
    #save_total_limit=3,
    gradient_accumulation_steps=2,
    eval_accumulation_steps=1,
    do_eval=True,
    do_train=True,
    weight_decay=0.1,
    #logging_dir = "logs",
    logging_strategy="steps",
    logging_steps = 10,
    dataloader_drop_last=True,
    save_safetensors=False,
    adam_epsilon=1e-08,
    warmup_steps=100,
    seed=42,
    lr_scheduler_type='cosine',
    load_best_model_at_end = True,
    label_smoothing_factor=0.01,
)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer = tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]

)

trainer.train()