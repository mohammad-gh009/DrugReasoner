import ast
import json
from typing import Any, List

import autogen
import numpy as np
import pandas as pd
import requests  # type: ignore
from DeepPurpose.dataset import load_broad_repurposing_hub
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


# Function to load configuration file
def load_config(config_file="config.json"):
    with open(config_file, "r") as f:
        return json.load(f)


def save_dti_results(drugs, targets, result, path):
    """Save DTI results to CSV file

    Args:
        drugs (List[str]): List of drug names
        targets (List[str]): List of target names
        result (DTIScore): DTI score results
        path (str): Path to save CSV file
    """
    # Convert DTIScore object to DataFrame
    df = pd.DataFrame(
        {
            "drug": drugs,
            "gene": targets,
            "ml_score": result.ml_dti_scores,
            "kg_score": result.kg_dti_scores,
            "search_score": result.search_dti_scores,
            "final_score": result.final_dti_scores,
            "reasoning": result.reasoning,
        }
    )

    try:
        existing_result = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
        merged_result = pd.concat([existing_result, df]).drop_duplicates(
            subset=["drug", "gene"], keep="last"
        )
        merged_result.to_csv(path, index=False)
    except FileNotFoundError:
        df.to_csv(path, index=False)


def average_dti_scores(
    ml_results: List[List[str | float]],
    kg_results: List[List[str | float]],
    search_results: List[List[str | float]],
    path: str = "average_dti_scores.csv",
    save: bool = False,
) -> dict[str, Any]:
    """
    Calculate average DTI scores for each drug-gene combination

    Args:
        ml_results: Results from machine learning [[drug, gene, score, reason], ...]
        kg_results: Results from Knowledge Graph [[drug, gene, score, reason], ...]
        search_results: Results from search [[drug, gene, score, reason], ...]
        path: Path to save results CSV
        save: Whether to save results to CSV (default: False)

    Returns:
        List[List[str | float]]: Averaged DTI scores [[drug, gene, avg_score], ...]
    """
    # Convert to DataFrames
    df_ml = pd.DataFrame(
        ml_results, columns=["drug", "gene", "ml_score", "reasoning"]
    ).iloc[:, :3]
    df_kg = pd.DataFrame(
        kg_results, columns=["drug", "gene", "kg_score", "reasoning"]
    ).iloc[:, :3]
    df_search = pd.DataFrame(
        search_results, columns=["drug", "gene", "search_score", "reasoning"]
    ).iloc[:, :3]

    # Merge on drug and gene
    df_merged = df_ml.merge(df_kg, on=["drug", "gene"], how="outer").merge(
        df_search, on=["drug", "gene"], how="outer"
    )

    # Calculate average of scores
    score_columns = ["ml_score", "kg_score", "search_score"]
    df_merged["avg_score"] = df_merged[score_columns].mean(axis=1)

    # Format results
    result = df_merged

    if save:
        try:
            existing_result = pd.read_csv(path)
            merged_result = pd.concat([existing_result, result]).drop_duplicates(
                subset=["drug", "gene"], keep="last"
            )
            merged_result.to_csv(path, index=False)
        except FileNotFoundError:
            result.to_csv(path, index=False)

    return {"columns": result.columns.tolist(), "values": result.values.tolist()}


CONFIG_LIST = load_config()


def create_agent(name, system_message, llm_config=CONFIG_LIST):
    return autogen.AssistantAgent(
        name=name,
        system_message=system_message,
        llm_config=llm_config,
    )


def get_target_name_from_uniprot(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("genes", [{}])[0].get("geneName", {}).get("value")
    except requests.RequestException as e:
        print(f"Error retrieving gene name for UniProt ID {uniprot_id}: {e}")
        return None


def get_smiles_from_compound_name(compound_name):
    # First try to get SMILES from local file
    try:
        df = pd.read_csv(
            "../data/nsc_cid_smiles_class_name.csv", usecols=["NAME", "SMILES"]
        )
        # Look for matching compound name
        match = df[df["NAME"].str.lower() == compound_name.lower()]
        if not match.empty:
            return match.iloc[0]["SMILES"]
    except Exception as e:
        print(f"Error reading local SMILES data: {e}")

    # If not found locally, try PubChem API
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/CanonicalSMILES/JSON"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        smiles = (
            data.get("PropertyTable", {})
            .get("Properties", [{}])[0]
            .get("CanonicalSMILES")
        )
        return smiles
    except requests.RequestException as e:
        print(f"Error retrieving SMILES for compound name {compound_name}: {e}")
        return None


def get_sequence_from_target_name(target_name):
    url = (
        f"https://rest.uniprot.org/uniprotkb/search?query={target_name}&fields=sequence"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [{}])[0].get("sequence", {}).get("value")
    except requests.RequestException as e:
        print(f"Error retrieving sequence for target name {target_name}: {e}")
        return None


def get_compound_name(smiles):
    SAVE_PATH = "./saved_path"
    X_repurpose, drug_name, drug_cid = load_broad_repurposing_hub(SAVE_PATH)
    return drug_name[X_repurpose == smiles][0]


def calculate_binary_metrics(y_true, y_pred, y_pred_proba=None, decimals=4):
    """
    Calculate binary classification metrics and return as a simple pandas DataFrame.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    y_pred_proba : array-like, optional
        Predicted probabilities for the positive class
    decimals : int, default=4
        Number of decimal places to round to

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing metrics and their values
    """
    metrics = {}

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Core metrics
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred)
    metrics["Recall"] = recall_score(y_true, y_pred)
    metrics["Specificity"] = tn / (tn + fp)
    metrics["F1"] = f1_score(y_true, y_pred)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    # Error rates
    metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics["FNR"] = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Add probability metrics if proba is provided
    if y_pred_proba is not None:
        metrics["AUC-ROC"] = roc_auc_score(y_true, y_pred_proba)
        metrics["AUPRC"] = average_precision_score(y_true, y_pred_proba)
        metrics["Log Loss"] = log_loss(y_true, y_pred_proba)
        metrics["Brier Score"] = np.mean((y_pred_proba - y_true) ** 2)

    # Create DataFrame and round values
    metrics_df = pd.DataFrame(
        {"metric_name": list(metrics.keys()), "value": list(metrics.values())}
    )

    metrics_df["value"] = metrics_df["value"].round(decimals)

    return metrics_df.sort_values("metric_name").reset_index(drop=True)


def extract_last_dti_score(chat_history):
    for message in reversed(chat_history):
        content = message.get("content", "")
        if not isinstance(content, str):
            continue

        # Look for a string that resembles a Python array
        if "[" in content and "]" in content:
            try:
                # Extract the array part
                array_text = content[content.find("[") : content.rfind("]") + 1]
                # Directly evaluate the string
                result = ast.literal_eval(array_text)

                if (
                    isinstance(result, list)
                    and len(result) >= 2
                    and isinstance(result[0], list)
                ):
                    headers = result[0]  # Header row
                    data_rows = result[1:]  # Data rows (multiple rows)

                    # Get the index of the headers
                    col_indices = {
                        "ML": headers.index("ML"),
                        "KG": headers.index("KG"),
                        "Search": headers.index("Search"),
                        "final_score": headers.index("final_score"),
                        "final_reasoning": headers.index("final_reasoning"),
                    }

                    # Initialize lists to store scores
                    ml_scores = []
                    kg_scores = []
                    search_scores = []
                    final_scores = []
                    reasonings = []

                    # Extract scores from each data row
                    for row in data_rows:
                        ml_scores.append(float(row[col_indices["ML"]]))
                        kg_scores.append(float(row[col_indices["KG"]]))
                        search_scores.append(float(row[col_indices["Search"]]))
                        final_scores.append(float(row[col_indices["final_score"]]))
                        reasonings.append(str(row[col_indices["final_reasoning"]]))

                    return {
                        "ml_dti_scores": ml_scores,
                        "kg_dti_scores": kg_scores,
                        "search_dti_scores": search_scores,
                        "final_dti_scores": final_scores,
                        "reasoning": reasonings,
                    }

            except (ValueError, SyntaxError, AttributeError, IndexError) as e:
                print(f"Debug - Error: {str(e)}")
                continue

    return None


def get_smiles_from_kca_graph(drug):
    df = pd.read_csv(
        "../data/kca_graph.csv.gz", index_col=0, usecols=["Drug", "SMILES"]
    ).drop_duplicates()
    return df[df["Drug"] == drug].iloc[0]["SMILES"]


def get_seq_from_kca_graph(target):
    df = pd.read_csv(
        "../data/kca_graph.csv.gz", index_col=0, usecols=["Gene", "Seq"]
    ).drop_duplicates()
    return df[df["Gene"] == target].iloc[0]["Seq"]
