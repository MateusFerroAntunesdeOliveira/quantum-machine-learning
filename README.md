# Machine Learning in Neuroscience: ASD Diagnosis Support

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Manager](https://img.shields.io/badge/manager-uv-purple)
![Status](https://img.shields.io/badge/status-research-orange)

A computational pipeline developed as part of a **Master's Thesis in Electrical Engineering (Bioinformatics/AI)** at the **Federal University of Paraná (UFPR)**. 

This project applies advanced Machine Learning techniques to phenotypic data from the **ABIDE I (Autism Brain Imaging Data Exchange)** repository to aid in the objective diagnosis of Autism Spectrum Disorder (ASD).


## Project Background & Problem Statement

**Context:**
Autism Spectrum Disorder (ASD) is a neurodevelopmental condition characterized by deficits in social communication and restricted behavioral patterns. Current diagnosis is predominantly clinical (ADOS-2, ADI-R) but faces challenges regarding subjectivity and accessibility.

**The Problem:**
* **Subjectivity:** Behavioral assessments can vary between clinicians.
* **Late Diagnosis:** Delays in identification impact access to early interventions.
* **Complexity:** High heterogeneity of the spectrum and comorbidity with other conditions makes classification difficult.

**Objective:**
To develop a robust, reproducible computational pipeline that classifies ASD vs. Control subjects using phenotypic data. The project aims to identify key **biomarkers** using Explainable AI (XAI) to support clinical decision-making.


## Solution Architecture

The methodology follows a rigorous 6-step scientific pipeline:

1.  **Data Acquisition:** Utilization of the ABIDE I dataset via the Preprocessed Connectomes Project (PCP).
2.  **Preprocessing & Cleaning:**
    * Threshold-based column removal (handling missing data).
    * **Hybrid Imputation Strategy:** Using **MICE** (Multivariate Imputation by Chained Equations) for core clinical attributes and **KNN** for supporting attributes.
3.  **Feature Engineering:**
    * Correlation analysis (Pearson, Spearman, PPS).
    * Polynomial feature generation and PCA (Principal Component Analysis).
4.  **Modeling:**
    * Systematic comparison of classifiers: SVM, Random Forest, Gradient Boosting (XGBoost, LightGBM), and Ensembles (Voting/Stacking).
5.  **Optimization:**
    * Hyperparameter tuning using **Optuna** within a **Nested Cross-Validation** (k=10) framework to prevent data leakage.
6.  **Explainability (XAI):**
    * Interpretation of model decisions using **SHAP** (SHapley Additive exPlanations).


## Technology Stack

This project is built on a modern Data Science stack and managed by **`uv`**.

* **Core:** Python 3.12+
* **Package Manager:** `uv` (Astral)
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
* **Optimization:** `optuna`
* **Visualization & XAI:** `matplotlib`, `seaborn`, `shap`, `ppscore`


## Project Structure

The project uses a modular architecture to separate configuration, logic, and execution steps.

```text
.
├── data
│   ├── input       # Raw phenotypic data (Phenotypic_V1_0b_preprocessed1.csv)
│   └── output      # Processed datasets and plots
│   └── logs        # execution logs
├── src
│   ├── processing  # Core logic modules
│   │   ├── analysis.py    # Plotting and Correlation logic
│   │   ├── cleaning.py    # Threshold dropping and initial cleanup
│   │   └── imputation.py  # MICE and KNN imputation strategies
│   ├── shared      # Shared resources
│   │   ├── config.py      # Constants, paths, and column definitions
│   │   ├── logger.py      # Centralized logging configuration
│   │   └── utils.py       # IO helper functions
│   └── steps       # Executable pipeline steps
│       ├── 01_data_cleaning.py
│       └── 02_exploratory_analysis.py
├── pyproject.toml  # Project dependencies managed by uv
├── uv.lock
└── README.md
```


## Getting Started

Follow these instructions to set up the research environment on your local machine.

### Prerequisites

You must have `uv` installed. It handles the virtual environment and package installation with extreme speed.

```bash
# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MateusFerroAntunesdeOliveira/quantum-machine-learning.git
    cd quantum-machine-learning
    ```
2.  **Sync dependencies:** This command creates the virtual environment and installs all required packages defined in `pyproject.toml`.
    ```bash
    uv sync
    ```

### Running the Pipeline

The pipeline is divided into sequential steps. Run them using uv run:

Step 01: Cleaning & Imputation Generates data/output/imputed_data.csv.

```bash
uv run python -m src.steps.01_data_cleaning
```

Step 02: Exploratory Data Analysis (EDA) Generates correlation matrices and PPS heatmaps in data/output/.

```bash
uv run python -m src.steps.02_exploratory
```


## Roadmap

* [X] **Phase 1:** Data Acquisition & Exploratory Analysis.
* [X] **Phase 2:** Advanced Preprocessing (MICE Imputation pipeline).
* [ ] **Phase 3:** Feature Selection & Engineering (Current Focus).
* [ ] **Phase 4:** Model Training with Nested Cross-Validation.
* [ ] **Phase 5:** Hyperparameter Optimization (Optuna).
* [ ] **Phase 6:** Explainability Analysis (SHAP) & Final Reporting.

<br>

----


**Author:** Mateus Ferro Antunes de Oliveira, **M.Sc. Student** in Electrical Engineering - Bioinformatics/AI at **Universidade Federal do Paraná** (UFPR).

**Advisor:** Prof. Dr. Leandro dos Santos Coelho

All rights reserved © 2025
