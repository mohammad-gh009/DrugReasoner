# DrugReasoner: Interpretable Drug Approval Prediction with a Reasoning-augmented Language Model
---

[Logo]()

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/Moreza009/drug_approval_all_classes)
[![arXiv](https://img.shields.io/badge/arXiv-1234.5678-b31b1b.svg)](https://arxiv.org/abs/2508.18579)

**DrugReasoner** is an AI-powered system for predicting drug approval outcomes using reasoning-augmented Large Language Models (LLMs) and molecular feature analysis. By combining advanced machine learning with interpretable reasoning, DrugReasoner provides transparent predictions that can accelerate pharmaceutical research and development.

![Figure 1.pdf](/properties/Figure_1.png "Schematic representation of DrugReasoner development and assessment")

## ✨ Key Features

- **🤖 LLM-Powered Predictions**: Utilizes fine-tuned Llama model for drug approval prediction
- **🧬 Molecular Analysis**: Advanced SMILES-based molecular structure analysis
- **🔍 Interpretable Results**: Clear reasoning behind predictions for better decision-making
- **📊 Similarity Analysis**: Identifies similar approved/non-approved compounds for context
- **⚡ Flexible Inference**: Support for both single molecule and batch predictions

## 🛠️ Installation
-  To use **DrugReasoner**, you must first request access to the base model [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on Hugging Face by providing your contact information. Once access is granted, you can run DrugReasoner either through the command-line interface (CLI) or integrate it directly into your Python workflows.

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training and inference)
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone git@github.com:mohammad-gh009/DrugReasoner.git
   cd DrugReasoner
   ```

2. **Create and activate virtual environment**

   **Windows:**
   ```bash
   cd src
   python -m venv myenv
   myenv\Scripts\activate
   ```

   **Mac/Linux:**
   ```bash
   cd src
   python -m venv myenv
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to use


**Note:** GPU is required for inference. If unavailable, use our [Colab Notebook]([link-to-colab](https://colab.research.google.com/drive/16OKB5q7MZ6MhWv5Q1I0QByN6DSkqx6az?usp=sharing)).


#### CLI Inference
```bash
python inference.py \
    --smiles "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" "CC1=CC=C(C=C1)C(=O)O" \
    --output results.csv \
    --top-k 9 \
    --top-p 0.9 \
    --max-length 2048 \
    --temperature 1.0
```

#### Python API Usage
```python
from inference import DrugReasoner

predictor = DrugReasoner()

results = predictor.predict_molecules(
    smiles_list=["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"],
    save_path="results.csv",
    print_results=True,
    top_k=9,
    top_p=0.9,
    max_length=2048,
    temperature=1.0
)
```

## 📊 Dataset & Model

- **Dataset**: [![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-drug_approval_prediction-yellow)](https://huggingface.co/datasets/Moreza009/drug_approval_prediction)
- **Model**: [![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-Llama--DrugReasoner-orange)](https://huggingface.co/Moreza009/Llama-DrugReasoner)

## 📈 Performance

DrugReasoner demonstrates superior performance compared to traditional baseline models across multiple evaluation metrics. Detailed performance comparisons are available in our [paper](https://arxiv.org/abs/2508.18579).


## 📝 Citation

If you use DrugReasoner in your research, please cite our work:

```
@misc{ghaffarzadehesfahani2025drugreasonerinterpretabledrugapproval,
      title={DrugReasoner: Interpretable Drug Approval Prediction with a Reasoning-augmented Language Model}, 
      author={Mohammadreza Ghaffarzadeh-Esfahani and Ali Motahharynia* and Nahid Yousefian and Navid Mazrouei and Jafar Ghaisari and Yousof Gheisari},
      year={2025},
      eprint={2508.18579},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.18579}, 
}
```

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


---

<div align="center">
  <strong>Accelerating drug discovery through AI-powered predictions</strong>
  <br><br>
</div>
