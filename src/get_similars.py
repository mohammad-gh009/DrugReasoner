import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import pairwise_distances
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

# Load the saved model
boosted_model = xgb.Booster()
boosted_model.load_model("../models/embedder/best_model.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

molformer = AutoModel.from_pretrained(
    "ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
)


molformer.eval()


def embed_smiles(smiles_list, batch_size=64):
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, return_tensors="pt", truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = molformer(**inputs)
        mask = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .expand(outputs.last_hidden_state.size())
            .float()
        )
        summed = torch.sum(outputs.last_hidden_state * mask, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        batch_embeddings = (summed / summed_mask).cpu()
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings).numpy()


def get_molecule_properties(smiles):
    """
    Compute molecular properties and analyses for a given SMILES string, including structural alerts (PAINS/Brenk).

    Args:
        smiles (str): SMILES string of the molecule.

    Returns:
        dict: A dictionary containing all computed properties and analyses.
    """
    # Initialize the molecule
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string provided.")

    # Dictionary to store all properties
    properties = {}

    # Basic descriptors
    properties["Molecular Weight"] = round(Descriptors.MolWt(molecule), 2)
    properties["LogP"] = round(Crippen.MolLogP(molecule), 2)
    properties["Molecular Refractivity"] = round(Crippen.MolMR(molecule), 2)
    properties["TPSA"] = round(Descriptors.TPSA(molecule), 2)
    properties["Hydrogen Bond_Donors"] = Lipinski.NumHDonors(molecule)
    properties["Hydrogen_Bond Acceptors"] = Lipinski.NumHAcceptors(molecule)
    properties["Rotatable Bonds"] = Lipinski.NumRotatableBonds(molecule)
    properties["Chiral Centers"] = len(
        Chem.FindMolChiralCenters(molecule)
    )  # Count instead of list
    properties["Heavy Atoms"] = molecule.GetNumHeavyAtoms()
    properties["Formal Charge"] = Chem.rdmolops.GetFormalCharge(molecule)
    properties["Total Rings"] = molecule.GetRingInfo().NumRings()

    # # Structural alerts (PAINS/Brenk)
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog.FilterCatalog(params)
    # Get all matches
    matches = catalog.GetMatches(molecule)
    if matches:
        properties["Structural Alerts"] = [entry.GetDescription() for entry in matches]
    else:
        properties["Structural Alerts"] = "None"

    return properties


def find_similar_molecules_leaf(model, X, labels):
    """
    Args:
        model: Trained XGBoost model (must be xgboost.Booster or have get_booster())
        X: Input features (numpy array/pandas DataFrame)
        labels: Array/list of 'approved'/'unapproved' labels
    """
    # Convert model to booster if using sklearn API
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    else:
        booster = model

    # Get leaf embeddings
    leaves = booster.predict(xgb.DMatrix(X), pred_leaf=True)

    # Split data into approved/unapproved groups
    approved_mask = np.array([l == 1 for l in labels])
    unapproved_mask = ~approved_mask

    approved_leaves = leaves[approved_mask]
    unapproved_leaves = leaves[unapproved_mask]

    approved_indices = np.where(approved_mask)[0]
    unapproved_indices = np.where(unapproved_mask)[0]

    results = {}

    for idx in range(len(leaves)):
        current_leaf = leaves[idx].reshape(1, -1)
        current_label = labels[idx]

        if current_label == 1:
            pos = np.where(approved_indices == idx)[0][0]
            approved_leaves_filtered = np.delete(approved_leaves, pos, axis=0)
            approved_idx_filtered = np.delete(approved_indices, pos)

            approved_dists = pairwise_distances(
                current_leaf, approved_leaves_filtered, metric="hamming"
            )[0]

            unapproved_dists = pairwise_distances(
                current_leaf, unapproved_leaves, metric="hamming"
            )[0]

        else:
            pos = np.where(unapproved_indices == idx)[0][0]
            unapproved_leaves_filtered = np.delete(unapproved_leaves, pos, axis=0)
            unapproved_idx_filtered = np.delete(unapproved_indices, pos)

            unapproved_dists = pairwise_distances(
                current_leaf, unapproved_leaves_filtered, metric="hamming"
            )[0]

            approved_dists = pairwise_distances(
                current_leaf, approved_leaves, metric="hamming"
            )[0]

        top5_approved = (
            approved_indices[np.argsort(approved_dists)[:5]]
            if current_label == 0
            else approved_idx_filtered[np.argsort(approved_dists)[:5]]
        )

        top5_unapproved = (
            unapproved_indices[np.argsort(unapproved_dists)[:5]]
            if current_label == 1
            else unapproved_idx_filtered[np.argsort(unapproved_dists)[:5]]
        )

        results[idx] = {
            "approved_neighbors": top5_approved.tolist(),
            "unapproved_neighbors": top5_unapproved.tolist(),
        }

    return results


def get_most(similarity_results,train_df,  idx=["approved_neighbors", "unapproved_neighbors"]):
    most_app = []
    for count in range(len(similarity_results)):
        most_app_inner = []
        for i in similarity_results[count][f"{idx}"]:
            most_app_inner.append(
                get_molecule_properties(train_df.loc[i]["smiles"])
            )
        most_app.append(str(most_app_inner))
    return most_app


def find_similar_molecules_val(model, X, labels, Z):
    """
    Args:
        model: Trained XGBoost model (must be xgboost.Booster or have get_booster())
        X: Input features (numpy array/pandas DataFrame) for reference dataset
        labels: Array/list of labels (1=approved, 0=unapproved) for reference dataset
        Z: Input features (numpy array/pandas DataFrame) for query dataset

    Returns:
        Dictionary mapping each query sample index to its similar approved/unapproved molecules from X
    """
    # Convert model to booster if using sklearn API
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    else:
        booster = model

    # Get leaf embeddings for both datasets
    X_leaves = booster.predict(xgb.DMatrix(X), pred_leaf=True)
    Z_leaves = booster.predict(xgb.DMatrix(Z), pred_leaf=True)

    # Split reference data into approved/unapproved groups
    approved_mask = np.array([l == 1 for l in labels])
    unapproved_mask = ~approved_mask

    approved_leaves = X_leaves[approved_mask]
    unapproved_leaves = X_leaves[unapproved_mask]

    approved_indices = np.where(approved_mask)[0]
    unapproved_indices = np.where(unapproved_mask)[0]

    results = {}

    current_leaf = Z_leaves.reshape(1, -1)

    # Calculate distances to approved molecules in X
    approved_dists = pairwise_distances(
        current_leaf, approved_leaves, metric="hamming"
    )[0]

    # Calculate distances to unapproved molecules in X
    unapproved_dists = pairwise_distances(
        current_leaf, unapproved_leaves, metric="hamming"
    )[0]

    # Get top 5 indices for both classes
    top5_approved = approved_indices[np.argsort(approved_dists)[:5]]
    top5_unapproved = unapproved_indices[np.argsort(unapproved_dists)[:5]]

    results = {
        "approved_neighbors": top5_approved.tolist(),
        "unapproved_neighbors": top5_unapproved.tolist(),
    }

    return results


def get_most_one(val_one,train_df, idx=["approved_neighbors", "unapproved_neighbors"]):
    most = []
    for i in val_one[f"{idx}"]:
        most.append(get_molecule_properties(train_df.loc[i]["smiles"]))
    return most
