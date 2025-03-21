import pandas as pd
from step1 import *
from ImportsAndDatasets import *
from parse_drugbank_data  import parse_drugbank ,convert_properties , extract_smiles
from preprocess_df_for_smiles import *

df_drugbank = parse_drugbank("/home/u111169/wrkdir/mgh-project/datasets/drugbank_data/full_database.xml")
print("drug bank data parsed")

approved_df, not_approved_df , lookup_table = preprocess_df_for_smiles(df_drugbank)

df_properties = lookup_table[["name","description", "average_mass", "toxicity","groups", "labels" ,"approval_stat", "SMILES", "experimental_properties", "calculated_properties" ]]

df = pd.read_csv("/home/u111169/wrkdir/mgh-project/ChemAP/dataset/DrugApp/All_training_feature_vectors.csv")
df = df[["SMILES" , "Label"]].rename(columns={'Label': 'labels'})

train_df , val_df , test_df , dataset_train , dataset_df , dataset_test = train_valid_test_split(df)


from rdkit import Chem

def get_canonical_smiles(smiles):
    # Convert input SMILES to molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None  # Invalid SMILES
    
    # Generate canonical SMILES
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return canonical_smiles

train_df["SMILES"] = train_df["SMILES"].apply(lambda x: get_canonical_smiles(x))
val_df["SMILES"] = val_df["SMILES"].apply(lambda x: get_canonical_smiles(x))
test_df["SMILES"] = test_df["SMILES"].apply(lambda x: get_canonical_smiles(x))

df_properties = df_properties.dropna(subset=["SMILES"])
df_properties["SMILES"] = df_properties["SMILES"].apply(lambda x: get_canonical_smiles(x))

def get_dict_of_approved_and_unapproved(df, dataset_train, dataset="val" , number_of_simmilar_in_each=5):
    """
    """
    df["Dicts"] = None
    for inst in range(len(df)): 
        app_dict = {}
        most_app , most_nonapp , tot  = get_most_app_and_most_nonapp(inst, df, dataset_train, dataset , number_of_simmilar_in_each)
        app_dict["most_app"]=most_app 
        app_dict["most_nonapp"]=most_nonapp 
        df["Dicts"][inst] = app_dict
    return df

def get_value(df, smiles_value, column_name):
    """
    Retrieves a value from a DataFrame based on a SMILES value and column name.

    Args:
        df (pd.DataFrame): The DataFrame to search in.
        smiles_value (str): The SMILES value to search for.
        column_name (str): The name of the column to retrieve the value from.

    Returns:
        The value from the specified column and SMILES value, or None if not found.
    """
    row = df[df["SMILES"] == smiles_value].reset_index()
    
    if not row.empty:
        return row[column_name][0]
    return None

def get_list_of_most(most_app:pd.DataFrame , df_properties): 
    final_list_of_most = []
    for i in most_app["SMILES"]: 
        dic = {}
        dic["name"] = get_value(df_properties,i, "name")
        dic["simmilarity score"] = most_app.loc[most_app["SMILES"] == i, "sim"].values
        dic["approval status"] = get_value(df_properties,i, "approval_stat")
        dic["average mass"] = get_value(df_properties , i, "average_mass")
        dic["toxicity"] = get_value(df_properties , i, "toxicity") 
        dic["descriptions"] = get_value(df_properties , i, "description") 
        final_list_of_most.append(dic) 
    return final_list_of_most
        
def final_dataset_containing_drug_information_as_list_ready_for_prompt(df, df_properties):
    # Initialize the new columns with None
    df["list_of_most_approved_info_for_n_simmilar"] = None
    df["list_of_most_nonapproved_info_for_n_simmilar"] = None
    
    # Iterate over each row in the DataFrame
    for i in range(len(df)):
        list_of_most_approved_info = get_list_of_most(df["Dicts"][i]["most_app"], df_properties)
        list_of_most_nonapproved_info = get_list_of_most(df["Dicts"][i]["most_nonapp"], df_properties)
        df.at[i, "list_of_most_approved_info_for_n_simmilar"] = list_of_most_approved_info
        df.at[i, "list_of_most_nonapproved_info_for_n_simmilar"] = list_of_most_nonapproved_info
        
    return df

def prepare_final_dataset(train_df, dataset_train, df_properties, dataset="train", number_of_simmilar_in_each=5):
    """
    Prepare the final dataset containing drug information as a list ready for prompt.

    Args:
    - train_df (DataFrame): The training DataFrame.
    - dataset_train (object): The training dataset object.
    - df_properties (DataFrame): The DataFrame containing properties.
    - number_of_simmilar_in_each (int, optional): The number of similar items in each. Defaults to 5.

    Returns:
    - final_df (DataFrame): The final dataset containing drug information as a list ready for prompt.
    """
    a = get_dict_of_approved_and_unapproved(train_df, dataset_train, dataset, number_of_simmilar_in_each=number_of_simmilar_in_each)
    final_df = final_dataset_containing_drug_information_as_list_ready_for_prompt(a, df_properties)
    return final_df


#a = get_dict_of_approved_and_unapproved(train_df, dataset_train, dataset="train" , number_of_simmilar_in_each=5)
#final_df = final_dataset_containing_drug_information_as_list_ready_for_prompt(a , df_properties)

#final_df.to_csv("train_reason.csv" , index = False)


a = get_dict_of_approved_and_unapproved(val_df, dataset_train, dataset="val" , number_of_simmilar_in_each=5)
final_df = final_dataset_containing_drug_information_as_list_ready_for_prompt(a , df_properties)

final_df.to_csv("val_reason.csv" , index = False)


#a = get_dict_of_approved_and_unapproved(test_df, dataset_train, dataset="test" , number_of_simmilar_in_each=5)
#final_df = final_dataset_containing_drug_information_as_list_ready_for_prompt(a , df_properties)

#final_df.to_csv("test_reason.csv" , index = False)
