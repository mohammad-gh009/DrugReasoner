import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def plot_training_loss(file_path, n_val_step):
    """
    Generate a plot of training and evaluation loss from a JSON file.

    Args:
    file_path (str): Path to the JSON file containing the training log history.
    n: the validation step that is used in the 
    Returns:
    None
    """

    # Open and load the JSON file
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)

    # Extract log history from the JSON data
    data = json_data["log_history"]

    # Extract steps and losses from the data
    steps = [entry["step"] for entry in data if "step" in entry and "loss" in entry]
    steps_ev = [step for step in steps if step % n_val_step == 0]
    losses_t = [entry["loss"] for entry in data if "step" in entry and "loss" in entry]
    losses_e = [entry["eval_loss"] for entry in data if "step" in entry and "eval_loss" in entry]

    # Create a line plot
    plt.plot(steps, losses_t, marker="o", label="Training Loss")
    plt.plot(steps_ev, losses_e, marker="o", label="Evaluation Loss")

    # Set labels and title
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()

    # Save the plot to a file
    plt.savefig("plot.png", bbox_inches="tight")

    # Display the plot
    plt.show()

# # Example usage:
# plot_training_loss("/home/u111169/wrkdir/mgh-project/arxive/checkpoints-arxive/fine_tuning_chemberta_classic_way/trainer-not-custom/checkpoint-585/trainer_state.json")

def evaluate(y_true ,y_pred ): 
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)#, average='weighted'
    precision = precision_score(y_true, y_pred)#, average='weighted'
    recall = recall_score(y_true, y_pred)#, average='weighted'

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    # Calculate specificity (True Negative Rate)
    specificity = TN / (TN + FP)

    # Calculate NPV (Negative Predictive Value)
    NPV = TN / (TN + FN)

    auc = roc_auc_score(y_true, y_pred)

    results = {
        "F1 score": f1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
        "Specificity": specificity,
        # "NPV": NPV
    }
    return results





def evaluate_model(model_path, test_data, batch_size=8):
    """
    Evaluate a pre-trained model on a test dataset.

    Args:
    - model_path (str): Path to the pre-trained model.
    - test_data (Dataset): Test dataset to evaluate the model on.
    - batch_size (int): Batch size for processing the test data.

    Returns:
    - y_true (list): List of true labels.
    - y_pred (list): List of predicted labels.
    """
    # Load pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    def preprocess_and_predict(examples):
        inputs = tokenizer(examples['SMILES'], return_tensors='pt', padding=True, truncation=True)
        inputs = inputs.to(device)
        outputs = model(**inputs)
        predicted_class_idx = torch.argmax(outputs.logits)
        predicted_class_prob = torch.nn.functional.softmax(outputs.logits, dim=1)[0][predicted_class_idx]
        predicted_class_idx = predicted_class_idx.cpu()  # Move tensors to CPU
        predicted_class_prob = predicted_class_prob.cpu()
        return {
            'predicted_class': predicted_class_idx,
            'predicted_class_prob': predicted_class_prob
        }
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Make predictions on test data
    test = test_data.map(preprocess_and_predict, batch_size=batch_size, writer_batch_size=batch_size)
    results_df = test.to_pandas()

    # Extract true labels and predicted labels
    y_true = results_df['labels'].tolist()
    y_pred = results_df['predicted_class'].tolist()

    return y_true, y_pred

# # Example usage:
# model_path = "/home/u111169/wrkdir/mgh-project/arxive/checkpoints-arxive/fine_tuning_chemberta_classic_way/trainer-not-custom/checkpoint-585"
# test_data = tokenized_ex  # Assuming tokenized_ex is your test dataset
# y_true, y_pred = evaluate_model(model_path, test_data)
# print("True labels:", y_true)
# print("Predicted labels:", y_pred)