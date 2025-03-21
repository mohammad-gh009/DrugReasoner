import pandas as pd
import numpy as np
#from rdkit import Chem
#from rdkit.Chem import AllChem
from datasets import Dataset
import itertools
from sklearn.model_selection import train_test_split
import gc


print("import completed")

df0 = pd.read_csv("/home/u111169/wrkdir/mgh-project/ChemAP/dataset/DrugApp/All_training_feature_vectors.csv")

print("dataframe created")

df=df0[["SMILES",	"Label"]]

# def morgan_fn(smile):
#     mol = Chem.MolFromSmiles(smile)
#     morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=2048)
#     morgan = ''.join(str(int(bit)) for bit in morgan_fp)
#     return morgan

# df["morgan"] = df["SMILES"].map(morgan_fn)
# df=df[["morgan" , "Label"]]

print("morgan dataframe created")
test_size = 0.2
val_size = 0.5
train_df , temp = train_test_split(df , stratify = df.Label , test_size = test_size , random_state=1234)
test_df , val_df = train_test_split(temp , stratify = temp.Label , test_size = val_size , random_state=1234)


#-----------------------------------
# reset index
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


def triplet_maker(df):
    pos_df = df[df["Label"]==1]["SMILES"]
    neg_df = df[df["Label"]==0]["SMILES"]
    elements = pos_df.tolist()
    elements_neg = neg_df.tolist()
    tuple_length = 2 
    unique_tuples = list(itertools.combinations(elements, tuple_length))
    triplets = []
    x = 0
    for pos_tup in unique_tuples:
        for neg in elements_neg:
            triple = (pos_tup[0],pos_tup[1], neg)
            triplets.append(triple)
            x = x+ 1
            if x%1_000_000 == 0 :
                print(x)
                gc.collect()
    print("Loop finished !")
    df_f = pd.DataFrame(triplets, columns=['Column1', 'Column2' , 'Column3'])
    return df_f

#train_df_d = triplet_maker(train_df)
#train_dataset = Dataset.from_pandas(train_df_d)
#print("training triplets  created. ")
#train_dataset.save_to_disk("/home/u111169/wrkdir/mgh-project/datasets/training_triplet_smiles")
#print("datasets saved at the local destination. ")
#gc.collect()

val_df_d = triplet_maker(val_df)
val_dataset = Dataset.from_pandas(val_df_d)
print("validation triplets  created. ")
val_dataset.save_to_disk("/home/u111169/wrkdir/mgh-project/datasets/validation_triplet_smiles")
print("datasets saved at the local destination. ")
gc.collect()

print("Finished !")
