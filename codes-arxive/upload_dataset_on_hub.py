from datasets import load_dataset , load_from_disk

print("starting . . .")
dataset = load_from_disk("/home/u111169/wrkdir/mgh-project/datasets/training_triplet_smiles")

print("dataset loaded from disk. ")

dataset.push_to_hub("Moreza009/triplet_smiles")

print("dataset uploaded on hub")
