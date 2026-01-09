from pathlib import Path

# * Command Center for Project Settings, Paths, Parameters, and Column Definitions

# --- PROJECT STRUCTURE ---
# Base dir = three levels up from this file:
# quantum-machine-learning/src/utils/config.py. So, BASE_DIR = quantum-machine-learning/

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
LOG_DIR = DATA_DIR / "logs"

# Ensure dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- FILE PATHS ---
LOG_FILE = LOG_DIR / "execution.log"
RAW_DATA_FILE = INPUT_DIR / "Phenotypic_V1_0b_preprocessed1_from_shawon.csv"
DROPPED_COLS_FILE = OUTPUT_DIR / "01_dropped_columns_report.csv"
IMPUTED_DATA_FILE = OUTPUT_DIR / "01_imputed_data.csv"

MISSING_VALUES_DISTRIBUTION_RAW_PLOT = OUTPUT_DIR / "02_missing_values_distribution_raw.png"
MISSING_VALUES_DISTRIBUTION_IMPUTED_PLOT = OUTPUT_DIR / "02_missing_values_distribution_imputed.png"
MISSINGNESS_STRATIFIED_PLOT = OUTPUT_DIR / "02_missingness_stratified.png"
CLASS_BALANCE_PLOT = OUTPUT_DIR / "02_class_balance.png"
HEATMAP_PLOT_SPEARMAN = OUTPUT_DIR / "02_heatmap_spearman.png"
HEATMAP_PLOT_PEARSON = OUTPUT_DIR / "02_heatmap_pearson.png"
HEATMAP_PPS_FULL_PLOT = OUTPUT_DIR / "02_heatmap_pps_full.png"
HEATMAP_PPS_TARGET_PLOT = OUTPUT_DIR / "02_heatmap_pps_target_only.png"
CORR_MATRIX_SPEARMAN_CSV = OUTPUT_DIR / "02_correlation_matrix_spearman.csv"
CORR_MATRIX_PEARSON_CSV = OUTPUT_DIR / "02_correlation_matrix_pearson.csv"
CORR_MATRIX_PPS_CSV = OUTPUT_DIR / "02_pps_matrix.csv"
CORR_MATRIX_PPS_PIVOT_CSV = OUTPUT_DIR / "02_pps_matrix_target_only.csv"

FINAL_TRAIN_DATA_FILE = OUTPUT_DIR / "03_X_train.csv"
FINAL_TARGET_DATA_FILE = OUTPUT_DIR / "03_y_train.csv"
SELECTED_FEATURES_FILE = OUTPUT_DIR / "03_selected_features.json"
SELECTED_FEATURES_CORR_PLOT = OUTPUT_DIR / "03_selected_features_correlation.png"

MODEL_COMPARISON_RESULTS_FILE = OUTPUT_DIR / "04_model_comparison_results.csv"

# Will be concatenated with model name, e.g., "05_best_hyperparameters_LightGBM.json"
BEST_PARAMS_FILE = OUTPUT_DIR / "05_best_hyperparameters"

# --- ID HANDLING ---
# * Column to be renamed to ID (usually the index from raw CSV)
RAW_ID_COLUMN = 'Unnamed: 0' 
ID_COLUMN = 'ID'

# --- PARAMETERS ---
MISSING_THRESHOLDS = {
    'core': 0.7,               # 70% tolerance for critical clinical features. Accept up to 69% missing.
    'support': 0.3,            # 30% for supporting. Accept up to 29% missing.
    'rare': 0.3,               # 30% for rare/other. Accept up to 29% missing.
    'default': 0.5             # 50% for any other columns. Accept up to 49% missing.
}

# --- COLUMNS DEFINITIONS ---
# * Target Mapping (ABIDE: 1=Autism, 2=Control -> ML: 1=Pos, 0=Neg)
TARGET_COLUMN = 'DX_GROUP'     # Diagnostic Group (1=Autism, 2=Control)
TARGET_MAPPING = {1: 1, 2: 0}

# Column to indicate percentage of missing values in reports
PERCENT_MISSING_COLUMN = '% Missing'

# Variance Threshold (remove low variance features). 0.01 means remove features where 99% of values are the same
VARIANCE_THRESHOLD = 0.01

# Correlation Threshold (for removing multicollinearity)
CORRELATION_THRESHOLD = 0.95

# * Belongs to Step 1 - Initial columns to drop before any processing (identifiers, leakage, duplicates)
COLS_TO_DROP_INITIALLY = [
    'Unnamed: 0.1',            # Duplicate index column
    'X',                       # Unknown
    'SUB_ID',                  # ABIDE Subject Identifier
    'subject',                 # Subject Identifier
    'DSM_IV_TR'                # ! Data Leakage, because its a variable derived from the final diagnosis
]

# * Belongs to Step 1 - Main attributes list to be used during cleaning and processing - can define a diagnostic set
CORE_ATTRIBUTES = [
    TARGET_COLUMN,             # Diagnostic Group (1=Autism, 2=Control)

    # * Removed [2025-12-09] to prevent data leakage:
    # 'DSM_IV_TR',               # Diagnostic Group (0=none, 1=Autism, 3=Aspergers, 4=PDD-NOS)

    # ADI-R: Autism Diagnostic Interview-Revised
    'ADI_R_SOCIAL_TOTAL_A',    # Social (66% missing)
    'ADI_R_VERBAL_TOTAL_BV',   # Verbal (66% missing)
    'ADI_RRB_TOTAL_C',         # Repetitive behaviors (66% missing)
    'ADI_R_ONSET_TOTAL_D',     # Onset (73% missing)
    'ADI_R_RSRCH_RELIABLE',    # Reliability (65% missing)

    # ADOS: Autism Diagnostic Observation Schedule
    'ADOS_MODULE',             # Module applied (54% missing)
    'ADOS_TOTAL',              # Total (63% missing)
    'ADOS_COMM',               # Communication (65% missing)
    'ADOS_SOCIAL',             # Social (65% missing)
    'ADOS_STEREO_BEHAV',       # Stereotypies behaviors (70% missing)
    'ADOS_RSRCH_RELIABLE',     # Reliability (66% missing)
    
    # SRS - Social Responsiveness Scale
    'SRS_RAW_TOTAL',           # Total Score (67% missing)
]

# * Belongs to Step 1 - Supporting attributes with moderate missingness or relevance - can define context
SUPPORTING_ATTRIBUTES = [
    'AGE_AT_SCAN',             # Age at scan
    'SEX',                     # Gender (1=Male, 2=Female)
    'EYE_STATUS_AT_SCAN',      # Eye status (1=Open, 2=Closed)
    'HANDEDNESS_CATEGORY',     # Handedness (R=Right, L=Left)
    'SITE_ID',                 # Site Identifier
    'SUB_IN_SMP',

    'FIQ',                     # Full IQ Standard Score (6% missing)
    'VIQ',                     # Verbal IQ Standard Score (17% missing)
    'PIQ',                     # Perfomance IQ Standard Score (16% missing)
    'FIQ_TEST_TYPE',           # IQ Test Used for Full Scale IQ (15% missing)
    'VIQ_TEST_TYPE',           # IQ Test Used for Verbal IQ (25% missing)
    'PIQ_TEST_TYPE',           # IQ Test Used for Perfomance IQ (24% missing)

    # Clinical Situation
    'CURRENT_MED_STATUS',      # Current medication status (27% missing)

    # Anatomic Data
    'anat_cnr','anat_efc','anat_fber','anat_fwhm','anat_qi1','anat_snr',

    # Functional Quality & Motion
    'func_efc','func_fber','func_fwhm','func_dvars','func_outlier','func_quality',
    'func_mean_fd','func_num_fd','func_perc_fd','func_gsr',

    # Manual QC (not text notes)
    'qc_rater_1','qc_anat_rater_2','qc_func_rater_2','qc_anat_rater_3','qc_func_rater_3'
]

# * Belongs to Step 1 - Rare attributes with high missingness but potential niche value
RARE_ATTRIBUTES = [
    # Secondary screening
    'SCQ_TOTAL',               # Social Communication Questionnaire (88% missing)
    'AQ_TOTAL',                # Autism Quotient Total Raw Score (95% missing)

    # Detailed subscales of SRS (Social Responsiveness Scale)
    'SRS_AWARENESS','SRS_COGNITION','SRS_COMMUNICATION','SRS_MOTIVATION','SRS_MANNERISMS',

    # QC (Quality Control) notes (textual)
    'qc_notes_rater_1','qc_anat_notes_rater_2','qc_func_notes_rater_2',
    'qc_anat_notes_rater_3','qc_func_notes_rater_3',
    'MEDICATION_NAME','COMORBIDITY','OFF_STIMULANTS_AT_SCAN'
]

# * Belongs to Step 2 - Critical features to check stratified missingness during EDA
CRITICAL_FEATURES = [
    'ADOS_TOTAL',
    'ADI_R_SOCIAL_TOTAL_A',
    'SRS_RAW_TOTAL',
    'ADOS_MODULE'
]

# * Belongs to Step 3 - Used for polynomial feature generation - Degree 2 + Interaction and based on PPS analysis
POLYNOMIAL_ATTRIBUTES = [
    # 1. ADOS_TOTAL             -> The main observational metric available for all patients.
    # 2. ADI_R_SOCIAL_TOTAL_A   -> Strong history-based social metric.
    # 3. SRS_RAW_TOTAL          -> Strong responsiveness scale metric.
    # 4. AGE_AT_SCAN & FIQ      -> Modulators for interactions.

    'ADOS_TOTAL',
    'ADI_R_SOCIAL_TOTAL_A',
    'SRS_RAW_TOTAL',
    'AGE_AT_SCAN',
    'FIQ',

    # ! BlindSpot
    # AGED_AT_SCAN and FIQ have a bad PPS correlation with the target alone.
    # They can not predict autism well (makes sense to the biological context).
    # However, a high ADOS score with 5 years is completely different than a high ADOS score at 25 years.
]
