import pandas as pd

from parse_drugbank_data  import parse_drugbank ,convert_properties , extract_smiles

def preprocess_df_for_smiles(df): 
    df_f = df[df["type"]=='small molecule']
    df_f.reset_index(inplace=True)

    # Handle both columns safely
    cols = ["experimental_properties", "calculated_properties"]
    for col in cols:
        df_f[col] = df_f[col].apply(convert_properties)

    # Apply the function to the entire column and create a new column
    df_f["SMILES"] = df_f["calculated_properties"].apply(extract_smiles)

    df_f.dropna(subset=['SMILES'], inplace=True)

    approved_df = df_f[df_f["groups"].apply(lambda x: 'approved' in x)]
    approved_df["approval_stat"] = "approved"

    not_approved_df = df_f[df_f["groups"].apply(lambda x: 'approved' not in x)]
    not_approved_df["approval_stat"] = "not_approved"

    df_cache_concat = pd.concat([approved_df , not_approved_df] , axis=0)
    
    return approved_df, not_approved_df , df_cache_concat



# ########################## df to train the embedding model ##########################
# df_to_train_embedding_model = df_cache_concat[["approval_stat", "groups","SMILES"]]

# df_to_train_embedding_model.to_csv("df_to_train_embedding_model.csv", index = False)