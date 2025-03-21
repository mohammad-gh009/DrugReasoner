from ImportsAndDatasets import *



# train_df , val_df , test_df , dataset_train , dataset_df , dataset_test = train_valid_test_split("/home/u111169/wrkdir/mgh-project/dataframes/the_drugbank_daatset_to_replace_with_chemap_for_test.csv")


def preprocess_and_extract_embeddings(examples):
    
    """
    in: smiles strings for each instances 
    fn: this will be used on the HF dataset.
    out: the full embedding of the last hidden state. 
    """
    
    inputs = tokenizer(examples['SMILES'], return_tensors='pt', padding=True, truncation=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state  # This is the embedding from the last layer
        # Take the average of the last hidden state across the sequence length dimension
        embeddings = last_hidden_state.mean(dim=1).squeeze()
        embeddings = embeddings.cpu()  # Move tensors to CPU
    return {
        'last_hidden_state':last_hidden_state, 
        'embeddings': embeddings.numpy(),
        'SMILES': examples['SMILES']  # Optional: include the original SMILES if needed
    }

def one_embeddings(examples):
    """
    in: one smile
    fn: get the last hidden state for one smile using the model
    out: full embedding of one smile
    """
    
    inputs = tokenizer(examples, return_tensors='pt', padding=True, truncation=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state  # This is the embedding from the last layer
        # Take the average of the last hidden state across the sequence length dimension
        embeddings = last_hidden_state.mean(dim=1).squeeze()
        embeddings = embeddings.cpu()  # Move tensors to CPU
    return {
        #'last_hidden_state':last_hidden_state,
        'embeddings': embeddings.numpy(),
        # 'SMILES': examples['SMILES']  # Optional: include the original SMILES if needed
    }
def get_sim_score(dataset_train, smile:str , app:int , nonapp:int )-> pd.DataFrame:
    """
    in: 
        dataset_train: the traininag dataframe 
        smile: the input smile that we want to get the simmilarity 
        app: number of simmilar approved drugs that we want to retreive. 
        nonapp: number of simmilar unapproved drugs that we want to retreive. 
    fn: 
    out: 
    """
    train_embeded= dataset_train.map(preprocess_and_extract_embeddings, batch_size=8, writer_batch_size=8).to_pandas() 
    main = torch.tensor(train_embeded["embeddings"])
    one = torch.tensor(one_embeddings(smile)["embeddings"])
    similarities = torch.cosine_similarity(one.unsqueeze(0), main, dim=1) 
    train_embeded["sim"]=similarities.tolist() 
    train_embeded.drop(columns=["embeddings" , "last_hidden_state"] , inplace = True)
    most_app = train_embeded[train_embeded["labels"]==1].sort_values(by=["sim"] , ascending = False)[:app]
    most_nonapp = train_embeded[train_embeded["labels"]==0].sort_values(by=["sim"] , ascending = False)[:nonapp]
    final = train_embeded.sort_values(by=["sim"] , ascending = False)[:nonapp]
    print(f"most simmilar SMILES to {smile} based on the model's prediction")
    return most_app , most_nonapp , final

def preprocess_and_predict_encoder(examples):
    """
    in: 
    fn: 
    out: 
    """
    inputs = tokenizer(examples, return_tensors='pt', padding=True, truncation = True)
    inputs = inputs.to(device)
    outputs = model(**inputs)
    predicted_class_idx = torch.argmax(outputs.logits)
    predicted_class_prob = torch.nn.functional.softmax(outputs.logits, dim=1)[0][predicted_class_idx]
    predicted_class_idx = predicted_class_idx.cpu()  # Move tensors to CPU
    predicted_class_prob = predicted_class_prob.cpu()
    return {
        'predicted_class_base': predicted_class_idx.item(),
        'predicted_class_prob_base': predicted_class_prob.item()
    }

def convert_to_fingerprint_for_eval(smiles):
    """
    in: 
    fn: 
    out: 
    """
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

    return target
def preprocess_and_predict_dencoder(examples):
    """
    in: 
    fn: 
    out: 
    """
    inputs = tokenizer(examples, return_tensors='pt', padding=True, truncation = True)
    inputs = inputs.to(device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=-1).item()
    predicted_label = tokenizer.decode(predicted_label)
    if predicted_label == "<Approved>":
        predicted_label = 1
    else: 
        predicted_label = 0

    return {
        'predicted_class': predicted_label,
        #'predicted_class_prob': probs
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path2model = "/home/u111169/wrkdir/mgh-project/arxive/checkpoints-arxive/fine_tuning_chemberta_classic_way/augmented/checkpoint-500"#'FacebookAI/xlm-roberta-large'

# # n = 112
# # smile_for_test = dataset_test["SMILES"][n]
# # labels_for_test=dataset_test["labels"][n]

model = AutoModel.from_pretrained(path2model).to(device)
tokenizer = AutoTokenizer.from_pretrained(path2model)
# most_app , most_nonapp , tot = get_sim_score(dataset_train , smile_for_test, 5,5)


def get_most_app_and_most_nonapp(index, df, dataset_train, dataset , number_of_simmilar_in_each):
    """
    
    """
    
    if dataset== "val": 
        smile_for_test = df["SMILES"][index]
        labels_for_test= df["labels"][index]   
        most_app , most_nonapp , tot = get_sim_score(dataset_train,smile_for_test, number_of_simmilar_in_each , number_of_simmilar_in_each) 

        return most_app , most_nonapp , tot
    
    elif dataset== "test": 
        smile_for_test = df["SMILES"][index]
        labels_for_test= df["labels"][index]   
        most_app , most_nonapp , tot = get_sim_score(dataset_train,smile_for_test, number_of_simmilar_in_each , number_of_simmilar_in_each) 

        return most_app , most_nonapp , tot    
    
    elif dataset== "train": 
        smile_for_test = df["SMILES"][index]
        labels_for_test= df["labels"][index] 
        dataset_train_for_embeding_the_rest_of_training_data = dataset_train.filter(lambda example: example["SMILES"] != smile_for_test)
        most_app , most_nonapp , tot = get_sim_score(dataset_train_for_embeding_the_rest_of_training_data,smile_for_test, number_of_simmilar_in_each , number_of_simmilar_in_each) 

        return most_app , most_nonapp , tot  
    else:
        raise ValueError("Invalid dataset type. Choose from 'val', 'test', or 'train'.")
        
# model = AutoModelForSequenceClassification.from_pretrained("/home/u111169/wrkdir/mgh-project/checkpoints-arxive/fine_tuning_chemberta_classic_way/augmented/checkpoint-500" ,num_labels=2).to(device)
# tokenizer = RobertaTokenizer.from_pretrained("/home/u111169/wrkdir/mgh-project/checkpoints-arxive/fine_tuning_chemberta_classic_way/augmented/checkpoint-500")#'FacebookAI/xlm-roberta-large'
# predicted = preprocess_and_predict_encoder(smile_for_test)




######################################################          GPT   ######################################################
# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir="/home/u111169/wrkdir/mgh-project/models")
# num_added_toks = tokenizer.add_tokens(["<Approved>", "<NotApproved>"])
# num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": ["<[lbl]>", "<[mr]>"]})
# print("We have added", num_added_toks, "tokens")
# model.resize_token_embeddings(len(tokenizer))



# # train_df["labels"] = train_df["labels"].map(convert_labels)
# val_df["SMILES"] = val_df.apply(lambda row: convert_to_fingerprint_for_eval(row["SMILES"]), axis=1)
# val_df["labels"] = val_df["labels"].map(convert_labels)
# test_df["SMILES"] = test_df.apply(lambda row: convert_to_fingerprint_for_eval(row["SMILES"]), axis=1)
# test_df["labels"] = test_df["labels"].map(convert_labels)

# n = 120
# smile_for_test = val_df["SMILES"][n]
# labels_for_test=val_df["labels"][n]

# path2model = "/home/u111169/wrkdir/mgh-project/checkpoints/druggpt2/checkpoint-9000"


# model = AutoModelForCausalLM.from_pretrained(path2model, is_decoder=True)# ,num_labels=2)
# model.to(device)
# model.eval()

# # for gpt2
# tokenizer.pad_token = tokenizer.eos_token
# predicted = preprocess_and_predict_dencoder(val_df["SMILES"][12])