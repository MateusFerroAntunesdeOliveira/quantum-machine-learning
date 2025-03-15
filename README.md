# Neuroimaging & Machine Learning

## Project Overview
This project explores the intersection of neuroimaging and machine learning. The primary goal is to preprocess and analyze anatomical and functional brain imaging data using classical machine learning (ML) techniques and quantum machine learning (QML). The project is part of a research initiative linked to a Master's program in Artificial Intelligence and Machine Learning applied to Neuroscience.

## Data Structure
The dataset consists of neuroimaging data collected from participants, including both anatomical and functional MRI scans. The data is organized as follows:

### **1. Data Folder (`data/`):**
This folder contains the neuroimaging data of participants, structured as follows:

- **`download_folder/NeuroIMAGE/`**
  - Contains multiple subdirectories named in the format `sub-XX`, where `XX` is the participant ID.
  - Each participant's folder follows this structure:
    
    - **Anatomical Data (`anat/`):**
      - Contains T1-weighted anatomical MRI scans in NIfTI format (`.nii.gz`).
      - Example file: `sub-0027003_ses-1_run-1_T1w.nii.gz`.
      
    - **Functional Data (`func/`):**
      - Contains functional MRI (fMRI) scans collected during resting-state tasks.
      - Example file: `sub-0027003_ses-1_task-rest_run-1_bold.nii.gz`.
      
    - **Participant Information (`participants.tsv`):**
      - A tab-separated values file containing metadata about participants, including:
        - `participant_id`: Unique identifier.
        - `gender`: Male/Female.
        - `age`: Age in years.
        - `handedness`: Left/Right-handed.
        - `dx`: Diagnosis (e.g., ADHD-Combined, Control).
        - `iq_measure`: IQ measurement method.
        - `full2_iq`: IQ score.
        - `qc_rest_1`: Quality control status for resting-state fMRI.
        - `qc_anatomical_1`: Quality control status for anatomical MRI.
        - `disclaimer`: Data usage restrictions.
      
    - **Additional Metadata (`T1w.json`):**
      - Contains details about MRI acquisition procedures.

### **2. Source Code (`src/`):**
This folder contains the scripts necessary for processing and analyzing the neuroimaging data.
- `main.py`: The main script for data loading, preprocessing, and model training.

## Project Workflow
The following steps outline the planned workflow for this project:

1. **Data Acquisition:**
   - Download and structure the neuroimaging dataset (completed).

2. **Data Preprocessing:**
   - Convert NIfTI images into numerical arrays.
   - Perform normalization and feature extraction.

3. **Machine Learning Pipeline:**
   - Apply classical ML models to classify ADHD vs. control subjects.
   - Train, validate, and optimize ML algorithms.

4. **Quantum Machine Learning (QML) Implementation:**
   - Reapply the dataset to QML models using Qiskit and other quantum libraries.
   - Compare the performance of classical ML and QML.

5. **Results Analysis & Documentation:**
   - Evaluate accuracy, efficiency, and interpretability of models.
   - Prepare a research article for submission to an Elsevier journal using Overleaf.

## Dependencies & Setup
Ensure the following Python packages are installed:
```bash
pip install -r requirements.txt
```

To run the main processing script:
```bash
python src/main.py
```

## Future Work
- Further fine-tuning of classical and quantum models.
- Extending the dataset to include additional neuroimaging sources.
- Exploring deep learning techniques for enhanced pattern recognition.

---
This research contributes to advancing AI applications in neuroscience by leveraging cutting-edge ML and QML techniques to analyze brain imaging data.

