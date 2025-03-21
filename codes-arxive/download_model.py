from sentence_transformers import SentenceTransformer , SentenceTransformerTrainer

print("import completed")

model = SentenceTransformer("DeepChem/ChemBERTa-77m-MTR" , cache_folder="/home/u111169/wrkdir/mgh-project/models")

print("Model downloaded !")
