import pandas as pd
from step1 import *
from ImportsAndDatasets import *
from parse_drugbank_data  import parse_drugbank ,convert_properties , extract_smiles
from preprocess_df_for_smiles import *

### parse the drugbank xml and return the dataframe of desired features 
df = parse_drugbank("/home/u111169/wrkdir/mgh-project/datasets/drugbank_data/full_database.xml")

approved_df, not_approved_df , lookup_table = preprocess_df_for_smiles(df)

df_properties = lookup_table[["name","description", "average_mass", "toxicity","groups", "labels" ,"approval_stat", "SMILES", "experimental_properties", "calculated_properties" ]]

train_df , val_df , test_df , dataset_train , dataset_df , dataset_test = train_valid_test_split(df_properties)


### pay attention that for testing I just use the 5 first of these dataframes 
train_df = train_df[: 5]
val_df = val_df[: 5]
test_df = test_df[: 5]
############################################################################

def get_dict_of_approved_and_unapproved(df, dataset_train, dataset="val" , number_of_simmilar_in_each=5):
    """
    Returns a modified dataframe with an additional column 'Dicts' containing dictionaries of approved and non-approved instances.
    
    Each dictionary in the 'Dicts' column contains two keys: 'most_app' and 'most_nonapp', which represent the most similar approved and non-approved instances, respectively.
    
    Args:
        df (pandas DataFrame): The input dataframe.
        dataset_train: The training dataset.
        dataset (str, optional): The dataset to use for finding similar instances. Defaults to "val".
        number_of_simmilar_in_each (int, optional): The number of similar instances to find for each type (approved and non-approved). Defaults to 5.
    
    Returns:
        pandas DataFrame: The modified dataframe with the additional 'Dicts' column.
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
    """
    gets the list of (for example 5 instances in each df) features with the explanations for each drug. 
    
    Args: 
        most_app: which can be the list of most approved or unapproved as dataframe 
        df_properties: this is the lookup table which we just select features from the main lookup table. 
    Returns: 
        A list which contatins all features with their explanations to use in the prompt. 
        
        ######## ATTENTION  ########
        the dict in the fucntion is manually designed, meaning if you want to consider more features from the lookup table OR the df_properties you should add the feature manually to the dict. 
    """
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
    df["list_of_most_approved_info"] = None
    df["list_of_most_nonapproved_info"] = None
    
    # Iterate over each row in the DataFrame
    for i in range(len(df)):
        list_of_most_approved_info = get_list_of_most(df["Dicts"][i]["most_app"], df_properties)
        list_of_most_nonapproved_info = get_list_of_most(df["Dicts"][i]["most_nonapp"], df_properties)
        df.at[i, "list_of_most_approved_info_for_n_simmilar"] = list_of_most_approved_info
        df.at[i, "list_of_most_nonapproved_info_for_n_simmilar"] = list_of_most_nonapproved_info
        
    return df


##############  The raw code for the next function 
# a = get_dict_of_approved_and_unapproved(train_df, dataset_train, dataset="train" , number_of_simmilar_in_each=5)
# final_df = final_dataset_containing_drug_information_as_list_ready_for_prompt(a , df_properties)

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



######################################################################################################################################################
#####################################################################   PROMPT   #####################################################################
######################################################################################################################################################

# sys_prompt = "you are an expert chemist "

# input_prompt_2 = f"""{sys_prompt}
# Your task is to provide a detailed analysis explaining why compound X has an approval status of {final_df["approval_stat"][0]}.

# I have developed a model that identifies molecules most similar to compound X. This model outputs two lists:
# 1. The five most similar approved small molecules (with their properties).
# 2. The five most similar non-approved small molecules (with their properties).

# Using the provided data:
# - Approved molecules: {final_df["list_of_most_approved_info"][0]}
# - Non-approved molecules: {final_df["list_of_most_nonapproved_info"][0]}

# Please analyze the data and reason which properties or patterns contribute to compound X being {final_df["approval_stat"][0]}. In your explanation, compare the characteristics observed in both lists to support your reasoning.
# """

