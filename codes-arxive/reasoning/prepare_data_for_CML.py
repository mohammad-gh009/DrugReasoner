########################## df to train the embedding model ##########################
df_to_train_embedding_model =pd.read_csv("/home/u111169/wrkdir/mgh-project/df_to_train_embedding_model.csv")

df_to_train_embedding_model=df_to_train_embedding_model.drop(columns = ["groups"])

df_to_train_embedding_model['approval_stat'] = df_to_train_embedding_model['approval_stat'].replace({'approved': 1, 'not_approved': 0})
df_to_train_embedding_model.rename(columns={"approval_stat":"labels"}, inplace=True)


df_to_train_embedding_model.to_csv("the_drugbank_daatset_to_replace_with_chemap_for_test.csv")