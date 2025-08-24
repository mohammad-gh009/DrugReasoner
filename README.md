# DrugReasoner: Interpretable Drug Approval Prediction with a Reasoning-augmented Language Model
---


[Logo]()

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/Moreza009/drug_approval_all_classes)

**DrugReasoner** is an AI-powered system that predicts drug approval outcomes using Large Language Models (LLMs) and molecular structure analysis. By combining advanced machine learning with interpretable reasoning, DrugReasoner helps accelerate pharmaceutical research and development.

## 📖 Abstract

Drug discovery is a complex and resource-intensive process, making early prediction of approval outcomes critical for optimizing research investments. While classical machine learning and deep learning methods have shown promise in drug approval prediction, their limited interpretability constraints their impact. Here, we present DrugReasoner, a reasoning-based large language model (LLM) built on the LLaMA architecture and fine-tuned with group relative policy optimization (GRPO) to predict the likelihood of small-molecule approval. DrugReasoner integrates molecular descriptors with comparative reasoning against structurally similar approved and unapproved compounds, generating predictions alongside step-by-step rationales and confidence scores. DrugReasoner achieved robust performance with an AUC of 0.732 and an F1 score of 0.729 on the validation set and 0.725 and 0.718 on the test set, respectively. These results outperformed conventional baselines, including logistic regression, support vector machine, and k-nearest neighbors and had competitive performance relative to XGBoost. On an external independent dataset, DrugReasoner outperformed both baseline and the recently developed ChemAP model, achieving an AUC of 0.728 and an F1-score of 0.774, while maintaining high precision and balanced sensitivity, demonstrating robustness in real-world scenarios. These findings demonstrate that DrugReasoner not only delivers competitive predictive accuracy but also enhances transparency through its reasoning outputs, thereby addressing a key bottleneck in AI-assisted drug discovery. This study highlights the potential of reasoning-augmented LLMs as interpretable and effective tools for pharmaceutical decision-making.

[Figure 1.pdf](https://github.com/user-attachments/files/21957023/Figure.1.pdf "Schematic representation of DrugReasoner development and assessment")

## ✨ Key Features

- **🤖 LLM-Powered Predictions**: Utilizes fine-tuned Llama model for drug approval prediction
- **🧬 Molecular Analysis**: Advanced SMILES-based molecular structure analysis
- **🔍 Interpretable Results**: Clear reasoning behind predictions for better decision-making
- **📊 Similarity Analysis**: Identifies similar approved/non-approved compounds for context
- **⚡ Flexible Inference**: Support for both single molecule and batch predictions
- **🎯 Baseline Comparisons**: Includes multiple baseline models (XGBoost, KNN, SVM, Logistic Regression)

## 🛠️ Installation

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
   python -m venv myenv
   myenv\Scripts\activate
   ```

   **Mac/Linux:**
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start


**Note:** GPU is required for inference. If unavailable, use our [Colab Notebook]([link-to-colab](https://colab.research.google.com/drive/16OKB5q7MZ6MhWv5Q1I0QByN6DSkqx6az?usp=sharing)).

#### Batch Inference

**Data Requirements:**
- Place your dataset in the `datasets/` folder
- Ensure your CSV file contains columns named `SMILES` and `Label`
- SMILES column should contain molecular structure strings
- Label column should contain the ground truth labels (if available)

```bash
cd DrugReasoner/src
python batch_inference.py --input ../datasets/test_processed.csv --output ../outputs/results.csv
```

**Example dataset format:**
```csv
SMILES,Label
"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",1
"CC1=CC=C(C=C1)C(=O)O",0
"CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)CF3",1
```

#### Single Molecule Inference
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


#### Performance Evaluation

Evaluate model performance using a CSV file with `y_pred` and `y_true` columns:

```bash
python utils.py --evaluate "path_to_results.csv"
```

## 📊 Datasets

- **Processed Data**: [drug_approval_prediction](https://huggingface.co/datasets/Moreza009/drug_approval_prediction)

## 📈 Performance

DrugReasoner demonstrates superior performance compared to traditional baseline models across multiple evaluation metrics. Detailed performance comparisons are available in our [paper]().


## 📝 Citation

If you use DrugReasoner in your research, please cite our work:

```bibtex

```

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


---

<div align="center">
  <strong>"Accelerating drug discovery through AI-powered predictions"</strong>
  <br>
