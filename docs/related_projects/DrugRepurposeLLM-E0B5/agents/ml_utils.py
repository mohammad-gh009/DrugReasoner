from typing import List, Optional, Union

from cache import FileCache
from DeepPurpose import DTI as models
from utils import get_sequence_from_target_name, get_smiles_from_compound_name

# Define a type alias for clarity
ResultType = List[Union[str, float]]


def get_dti_score(drug_names: List[str], target_names: List[str]) -> List[ResultType]:
    cache = FileCache("ml_dti_scores")
    reasoning = "This agent used an ML model"
    result: List[Optional[ResultType]] = [None] * len(drug_names)
    uncached_pairs = []

    # Check the cache
    for i, (drug_name, target_name) in enumerate(zip(drug_names, target_names)):
        cache_key = f"{drug_name}_{target_name}"
        cached_value = cache.get(cache_key)

        if cached_value is not None:
            result[i] = [drug_name, target_name, cached_value, reasoning]
        else:
            drug = get_smiles_from_compound_name(drug_name)
            target = get_sequence_from_target_name(target_name)
            uncached_pairs.append((i, drug, target, drug_name, target_name))

    # Process uncached pairs
    if uncached_pairs:
        net = models.model_pretrained("MPNN_CNN_BDB")

        # Simplify the unpacking process
        indices = [p[0] for p in uncached_pairs]
        uncached_drugs = [p[1] for p in uncached_pairs]
        uncached_targets = [p[2] for p in uncached_pairs]
        uncached_drug_names = [p[3] for p in uncached_pairs]
        uncached_target_names = [p[4] for p in uncached_pairs]

        predictions = models.virtual_screening(
            uncached_drugs,
            uncached_targets,
            net,
            uncached_drug_names,
            uncached_target_names,
        )

        # Save the results
        for i, drug_name, target_name, probability in zip(
            indices, uncached_drug_names, uncached_target_names, predictions
        ):
            cache_key = f"{drug_name}_{target_name}"
            cache.set(cache_key, probability)
            result[i] = [drug_name, target_name, probability, reasoning]

    return [r for r in result if r is not None]
