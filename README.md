# Lightweight Intrusion Detection for IoT Networks

**CS 6140 — Machine Learning**
**Team:** Mahip Parekh & Piyush Daga

## Project Overview
Feature-efficient intrusion detection for IoT networks using the RT-IoT2022 dataset. We compare feature selection methods (Random Forest importance, PCA) and class imbalance strategies (SMOTE, cost-sensitive learning) across four classifiers: Logistic Regression, SVM, Random Forest, and DNN (PyTorch).

## Dataset
RT-IoT2022 from UCI ML Repository — 123,117 network flows, 83 features, 13 classes.
- Download: https://doi.org/10.24432/C5P338
- **Do not push dataset to Git** — keep in shared Google Drive

## Repo Structure
```
iot-ids-ml/
├── data/                    # .gitignored — dataset on Google Drive
├── notebooks/
│   ├── piyush/              # Piyush's notebooks
│   └── mahip/               # Mahip's notebooks
├── src/                     # Shared utility functions
│   ├── data_loader.py       # Data loading & exploration
│   ├── preprocessing.py     # Encoding, scaling, splitting, SMOTE
│   └── evaluation.py        # Metrics, confusion matrix, reports
├── results/                 # Experiment result CSVs
├── models/                  # Saved trained models
├── figures/                 # All plots and charts
└── README.md
```

## How to Run
1. Clone this repo
2. Download RT-IoT2022 dataset and place CSV in `data/` folder
3. Install dependencies: `pip install -r requirements.txt`
4. Run notebooks in order from `notebooks/`

## Models
- Logistic Regression
- SVM (RBF kernel)
- Random Forest
- DNN (PyTorch)

## Evaluation
- Primary metric: Macro-averaged F1
- Per-class precision, recall, F1 for all 13 classes
- Confusion matrices
