import argparse
import torch
import pandas as pd
from datasets import Dataset , load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from get_similars import *
import os
from datetime import datetime

dataset = load_dataset(
    "Moreza009/drug_approval_prediction",
    cache_dir="../datasets",
)

train_df = dataset["train"].to_pandas()
df_embed = embed_smiles(train_df["smiles"].tolist())


class DrugReasoner:
    """
    A class to handle drug discovery predictions using a fine-tuned Llama model.
    """
    
    def __init__(self, model_name="meta-llama/Llama-3.1-8B" ,#
                 peft_model="Moreza009/Llama-DrugReasoner" ):#
        self.model_name = model_name
        self.peft_model = peft_model
        self.tokenizer = None
        self.model = None
        self.system_prompt = """
You are a chemist specializing in drug discovery and molecular modeling. Your role is to evaluate the drug-likeness and viability of a given chemical compound as a potential drug candidate, using computational descriptors derived from RDKit.

For each compound, you are provided with:
- Molecular descriptors and properties calculated via RDKit.
- The RDKit-derived descriptors and properties of the five most similar approved and five most similar unapproved small molecules (based on structural similarity).

Follow these steps:
1. Analyze the compound's physicochemical properties.
2. Compare the compound with its most similar approved and unapproved counterparts, highlighting key similarities and differences that may influence drug-likeness or viability.
3. Think step by step and use your knowledge of molecular approval criteria and the patterns observed in both approved and unapproved compounds to explain your reasoning clearly and scientifically.

Based on your analysis, label the compound as either:
- approved
- unapproved

Then assign a confidence score between 0 and 1, where:
- A score near 1.0 indicates high confidence in your prediction (clear, strong evidence).
- A score near 0.0 indicates high uncertainty (ambiguous, conflicting, or weak evidence).

Respond in the following format:

<think>
(Your step-by-step scientific reasoning)
</think>
<label>
(approved or unapproved)
</label>
<score>
(Numeric confidence score between 0 and 1)
</score>
"""
        # Default generation args (will be overridden by parameters in predict_molecules)
        self.default_generation_args = {
            "temperature": 1.0,
            "max_length": 4096,
            "top_p": 0.9,
            "top_k": 9,
        }
        
    def load_model(self):
        """Load the tokenizer and model."""
        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            quantization_config=bnb_config, 
            device_map="auto"
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.peft_model).to("cuda")
        self.default_generation_args["pad_token_id"] = self.tokenizer.eos_token_id
        print("Model loaded successfully!")
        
    def prepare_molecule_data(self, smiles_list):
        """
        Prepare molecular data from SMILES strings.
        
        Args:
            smiles_list (list): List of SMILES strings
            
        Returns:
            pd.DataFrame: DataFrame with molecular properties and similar molecules
        """
        print(f"Processing {len(smiles_list)} molecules...")
        
        df_input = pd.DataFrame({"SMILES": smiles_list})
        
        # Calculate RDKit properties
        print("Calculating RDKit properties...")
        df_input["rdkit"] = df_input["smiles"].apply(
            lambda x: str(get_molecule_properties(x))
        )
        
        # Find most similar approved molecules
        print("Finding similar approved molecules...")
        df_input["most_app"] = df_input["smiles"].apply(
            lambda x: str(get_most_one(
                find_similar_molecules_val(
                    boosted_model, df_embed, train_df["label"], embed_smiles([x])
                ),train_df, 
                idx="approved_neighbors",
            ))
        )
        
        # Find most similar unapproved molecules
        print("Finding similar unapproved molecules...")
        df_input["most_nonapp"] = df_input["smiles"].apply(
            lambda x: str(get_most_one(
                find_similar_molecules_val(
                    boosted_model, df_embed, train_df["label"], embed_smiles([x])
                ),train_df, 
                idx="unapproved_neighbors",
            ))
        )
        
        return df_input
    
    def create_dataset(self, df_input):
        """
        Create a dataset from the prepared molecular data.
        
        Args:
            df_input (pd.DataFrame): DataFrame with molecular data
            
        Returns:
            Dataset: HuggingFace Dataset object
        """
        data = Dataset.from_pandas(df_input)
        data = data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"""
                Compound's RDKit Properties:{x["molecular_features"]} 

                Five Most Similar Approved Compounds (RDKit): {x['most_similar_approved']} 
                
                Five Most Similar Unapproved Compounds (RDKit): {x['most_similar_unapproved']}
                """,
                    },
                ],
                "SMILES": x["smiles"],
            }
        )
        return data
    
    def extract_xml_content(self, text, tag):
        """Extract content from XML-like tags."""
        try:
            return text.split(f"<{tag}>")[-1].split(f"</{tag}>")[0].strip()
        except:
            return "N/A"
    
    def predict_molecules(self, smiles_list, save_path="../outputs/results.csv", print_results=True,
                         top_k=9, top_p=0.9, max_length=4096, temperature=1.0):
        """
        Main function to predict drug-likeness for a list of SMILES.
        
        Args:
            smiles_list (list): List of SMILES strings
            save_path (str, optional): Path to save the results CSV
            print_results (bool): Whether to print the results DataFrame
            top_k (int): Top-k sampling parameter (default: 9)
            top_p (float): Top-p (nucleus) sampling parameter (default: 0.9)
            max_length (int): Maximum generation length (default: 4096)
            temperature (float): Sampling temperature (default: 1.0)
            
        Returns:
            pd.DataFrame: Results DataFrame with predictions
        """
        if self.model is None:
            self.load_model()
        
        # Set up generation arguments with provided parameters
        generation_args = self.default_generation_args.copy()
        generation_args.update({
            "top_k": top_k,
            "top_p": top_p,
            "max_length": max_length,
            "temperature": temperature,
        })
        
        print(f"Generation parameters: top_k={top_k}, top_p={top_p}, max_length={max_length}, temperature={temperature}")
            
        # Prepare molecular data
        df = self.prepare_molecule_data(smiles_list)
        
        # Create dataset
        dataset = self.create_dataset(df)
        
        # Make predictions
        print("Making predictions...")
        labels, scores, think = [], [], []
        
        for i in range(len(dataset)):
            print(f"Processing molecule {i+1}/{len(dataset)}")
            
            inputs = self.tokenizer.apply_chat_template(
                dataset["prompt"][i], 
                return_tensors="pt", 
                add_generation_prompt=True
            ).to("cuda")
            
            outputs = self.model.generate(inputs, **generation_args)
            input_length = inputs.shape[1]
            decoded = self.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            print(f"Model output for molecule {i+1}:")
            print(decoded)
            print("-" * 80)
            
            labels.append(self.extract_xml_content(decoded, "label").lower())
            scores.append(self.extract_xml_content(decoded, "score").lower())
            think.append(self.extract_xml_content(decoded, "think").lower())
        
        # Create output DataFrame
        out_df = pd.DataFrame({
            "SMILES": smiles_list,
            "thinking": think,
            "prediction": labels,
            "confidence_score": scores,
            "rdkit_properties": df["molecular_features"].tolist(),
            "similar_approved": df["most_similar_approved"].tolist(),
            "similar_unapproved": df["most_similar_unapproved"].tolist(),
        })
        
        # Print results if requested
        if print_results:
            print("\n" + "="*100)
            print("PREDICTION RESULTS")
            print("="*100)
            print(out_df.to_string(index=False, max_colwidth=50))
            print("="*100)
        
        # Save results if path provided
        if save_path:
            out_df.to_csv(save_path, index=False)
            print(f"\nResults saved to: {save_path}")
            
        return out_df


def main():
    """
    Main function to run the drug discovery prediction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Predict drug-likeness for small molecules"
    )
    parser.add_argument(
        "--smiles", 
        nargs="+", 
        required=True,
        help="List of SMILES strings to analyze"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output CSV file path (optional)"
    )
    parser.add_argument(
        "--no-print", 
        action="store_true",
        help="Don't print results to console"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=9,
        help="Top-k sampling parameter (default: 9)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum generation length (default: 4096)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DrugReasoner()
    
    # Generate default output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"drug_predictions_{timestamp}.csv"
    
    # Run predictions
    out_df = predictor.predict_molecules(
        smiles_list=args.smiles,
        save_path=args.output,
        print_results=not args.no_print,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    return out_df


# Example usage function
def example_usage():
    """
    Example of how to use the DrugDiscoveryPredictor class.
    """
    # Example SMILES strings (replace with your actual molecules)
    example_smiles = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC1=CC=C(C=C1)C(=O)O",           # p-Toluic acid
        "C1=CC=C(C=C1)C(=O)O"             # Benzoic acid
    ]
    
    # Initialize predictor
    predictor = DrugReasoner()
    
    # Make predictions
    results = predictor.predict_molecules(
        smiles_list=example_smiles,
        save_path="example_results.csv",
        print_results=True,
        top_k=9,
        top_p=0.9,
        max_length=2048,
        temperature=1
    )
    
    return results


if __name__ == "__main__":
    # Uncomment the line below to run the example
    # example_usage()
    
    # Run main function with command line arguments
    main()