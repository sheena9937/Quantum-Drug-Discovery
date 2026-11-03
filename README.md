# Quantum Drug Discovery: BBBP Classification

## Overview

This project explores the application of Quantum Machine Learning (QML) to molecular classification using the BBBP (Blood-Brain Barrier Penetration) dataset.

The objective is to investigate whether quantum machine learning models can classify molecules based on their ability to penetrate the blood-brain barrier and compare their performance with classical machine learning algorithms.

## Dataset

- Dataset: BBBP (Blood-Brain Barrier Penetration)
- Total molecules: 2,039
- Molecular representation: Morgan fingerprints
- Fingerprint dimensions: 1,024
- Class 0: Molecules that do not penetrate the blood-brain barrier
- Class 1: Molecules that penetrate the blood-brain barrier

## Methodology

The project follows these steps:

1. Load and preprocess the BBBP molecular dataset.
2. Generate 1,024-bit Morgan fingerprints using RDKit.
3. Split the dataset into training and testing subsets.
4. Apply PCA to reduce the feature dimensions from 1,024 to 4 for quantum experiments.
5. Apply scaling to prepare the reduced features for quantum circuits.
6. Train classical machine learning models.
7. Implement quantum machine learning models using Qiskit.
8. Compare model performance using multiple evaluation metrics.

## Classical Machine Learning Models

The following models were implemented:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- Class imbalance experiments

## Quantum Machine Learning Models

The following experiments were performed:

- Qiskit circuit fundamentals
- Quantum data preparation
- Quantum Support Vector Classifier (QSVC)
- Z Feature Map
- ZZ Feature Map
- Feature map repetition comparison
- Variational Quantum Classifier (VQC)
- VQC class imbalance experiment

## Final Pilot Comparison

The final comparison used 120 training samples and 100 testing samples for each reported model.

| Model | Accuracy | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.80 | 0.6261 | 0.7374 |
| Random Forest | 0.81 | 0.6327 | 0.7166 |
| SVM | 0.79 | 0.5768 | 0.7538 |
| Z Feature Map QSVC | 0.77 | 0.5779 | 0.7708 |
| ZZ QSVC | 0.76 | 0.6140 | 0.7253 |
| ZZ QSVC (2 Reps) | 0.72 | 0.6447 | 0.7160 |
| Original VQC | 0.65 | 0.4419 | 0.6754 |
| Balanced VQC | 0.61 | 0.4298 | 0.4161 |

## Key Findings

- Random Forest achieved the highest accuracy of 81%.
- Z Feature Map QSVC achieved the highest ROC-AUC of 0.7708.
- ZZ QSVC with two repetitions achieved the highest balanced accuracy of 0.6447.
- VQC experiments struggled with minority-class classification.
- Model performance depended on the selected evaluation metric and circuit configuration.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- RDKit
- Qiskit
- Qiskit Machine Learning
- Matplotlib
- XGBoost

## Project Structure

```text
Quantum-Drug-Discovery/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── figures/
│
├── results/
│   ├── comparisons/
│   ├── models/
│   └──
│
├── src/
│   ├── preprocessing/
│   ├── classical_models/
│   ├── experiments/
│   ├── optimization/
│   └── quantum_models/
│
├── research_notes
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/sheena9937/Quantum-Drug-Discovery.git
cd Quantum-Drug-Discovery
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Note: If required, install the compatible Qiskit Machine Learning and Qiskit Algorithms packages separately.

## Limitations

- Quantum experiments were conducted on small pilot subsets because of computational costs.
- PCA reduced the original 1,024 molecular features to four dimensions, resulting in information loss.
- VQC performance was limited, particularly for minority-class prediction.
- The experiments do not demonstrate a general quantum advantage.
- Further validation on larger datasets and repeated train/test splits is required.

## Future Scope

- Explore alternative dimensionality reduction techniques.
- Experiment with different quantum feature maps and ansatz configurations.
- Improve minority-class prediction.
- Evaluate quantum models on larger datasets.
- Investigate more efficient quantum kernel methods.

## Conclusion

This project implements and compares classical and quantum machine learning approaches for BBBP molecular classification.

The experiments provide practical insights into quantum feature maps, variational quantum circuits, molecular feature representation and model evaluation.

The results highlight both the potential and current limitations of applying quantum machine learning to drug discovery.

## Author

Developed as a Quantum Machine Learning project exploring molecular classification and quantum-enhanced learning methods.