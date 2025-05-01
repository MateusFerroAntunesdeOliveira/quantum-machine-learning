import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

CORE_ATTRIBUTES = [
    'DX_GROUP',                # Diagnostic Group (1=Autism, 2=Control)
    'DSM_IV_TR',               # Diagnostic Group (0=none, 1=Autism, 3=Aspergers, 4=PDD-NOS)

    # ADI-R: Autism Diagnostic Interview-Revised
    'ADI_R_SOCIAL_TOTAL_A',    # Social (63% missing)
    'ADI_R_VERBAL_TOTAL_BV',   # Verbal (63% missing)
    'ADI_RRB_TOTAL_C',         # Repetitive behaviors (63% missing)
    'ADI_R_ONSET_TOTAL_D',     # Onset (70% missing)
    'ADI_R_RSRCH_RELIABLE',    # Reliability (63% missing)

    # ADOS: Autism Diagnostic Observation Schedule
    'ADOS_MODULE',             # Module applied (52% missing)
    'ADOS_TOTAL',              # Total (60% missing)
    'ADOS_COMM',               # Communication (62.5% missing)
    'ADOS_SOCIAL',             # Social (62.5% missing)
    'ADOS_STEREO_BEHAV',       # Stereotypies behaviors (66% missing)
    'ADOS_RSRCH_RELIABLE',     # Reliability (66% missing)
    
    # SRS - Social Responsiveness Scale
    'SRS_RAW_TOTAL',           # Total Score (63% missing)
]

SUPPORTING_ATTRIBUTES = [
    'AGE_AT_SCAN',             # Age at scan
    'SEX',                     # Gender (1=Male, 2=Female)
    'EYE_STATUS_AT_SCAN',      # Eye status (1=Open, 2=Closed)
    'HANDEDNESS_CATEGORY',     # Handedness (R=Right, L=Left)
    'SITE_ID',
    'SUB_IN_SMP',

    'FIQ',                     # Full IQ Standard Score (3% missing)
    'VIQ',                     # Verbal IQ Standard Score (16%)
    'PIQ',                     # Perfomance IQ Standard Score (14%)
    'FIQ_TEST_TYPE',
    'VIQ_TEST_TYPE',
    'PIQ_TEST_TYPE',

    # Clinical Situation
    'CURRENT_MED_STATUS',      # Current medication status (26% missing)
    
    # Anatomic Data
    'anat_cnr','anat_efc','anat_fber','anat_fwhm','anat_qi1','anat_snr',
    
    # Functional Quality & Motion
    'func_efc','func_fber','func_fwhm','func_dvars','func_outlier','func_quality',
    'func_mean_fd','func_num_fd','func_perc_fd','func_gsr',
    
    # Manual QC (not text notes)
    'qc_rater_1','qc_anat_rater_2','qc_func_rater_2','qc_anat_rater_3','qc_func_rater_3'
]

RARE_ATTRIBUTES = [
    # Secondary screening
    'SCQ_TOTAL',               # Social Communication Questionnaire (87% missing)
    'AQ_TOTAL',                # Autism Quotient Total Raw Score (94% missing)

    # Detailed subscales of SRS (Social Responsiveness Scale)
    'SRS_AWARENESS','SRS_COGNITION','SRS_COMMUNICATION','SRS_MOTIVATION','SRS_MANNERISMS',

    # QC (Quality Control) notes (textual)
    'qc_notes_rater_1','qc_anat_notes_rater_2','qc_func_notes_rater_2',
    'qc_anat_notes_rater_3','qc_func_notes_rater_3',
    'MEDICATION_NAME','COMORBIDITY','OFF_STIMULANTS_AT_SCAN'
]

def load_data(path):
    print(f"\n[1] Loading data from: {path}")
    dataframe = pd.read_csv(path, sep=',')
    print(f"    Data shape: {dataframe.shape}")
    return dataframe

def missing_value_report(df):
    missing_value = df.isnull().sum().rename("Total Missing").to_frame()
    missing_value['% Missing'] = 100 * missing_value['Total Missing'] / len(df)
    missing_value = missing_value.sort_values('% Missing', ascending=False)
    missing_value.drop("Total Missing", axis=1, inplace=True)
    print('\n[2] Missing value summary:')
    print(missing_value.head())
    return missing_value

def plot_missing_distribution(missing_values, bins=20):
    """
    Plot histogram of % missing values across all features.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(missing_values['% Missing'], bins=bins, color='blue', alpha=0.7)
    plt.title('Distribution of Missingness Across Features')
    plt.xlabel('% Missing Values')
    plt.ylabel('Number of Columns')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def drop_by_missing_threshold(report, df, thresholds):
    """
    Conditionally drop columns based on thresholds dict:
    Returns cleaned df and list of dropped columns.
    """
    to_drop = []
    for col, pct in report['% Missing'].items():
        if col in CORE_ATTRIBUTES:
            limit = thresholds['core']
        elif col in SUPPORTING_ATTRIBUTES:
            limit = thresholds['support']
        elif col in RARE_ATTRIBUTES:
            limit = thresholds['rare']
        else:
            limit = thresholds['default']
        if pct/100 > limit:
            to_drop.append(col)

    print(f"\n[3] Conditionally dropping {len(to_drop)} columns based on thresholds: {to_drop}")
    df_clean = df.drop(columns=to_drop)
    print(f"\n    New shape after conditional drop: {df_clean.shape}")
    print(f"\n[3] Remaining columns: {df_clean.columns.tolist()}")
    return df_clean, to_drop

def impute_missing(df):
    """
    Advanced imputation for three groups of colunas:
      - CORE_ATTRIBUTES (all numeric): IterativeImputer with BayesianRidge MICE
      - SUPPORTING_ATTRIBUTES:
          - numeric: KNNImputer(n_neighbors=5)
          - categorical: SimpleImputer(strategy='most_frequent')
      - Default:
          - numeric: SimpleImputer(strategy='median')
          - categorical: SimpleImputer(strategy='most_frequent')
    """
    print('\n[4] Imputing missing values with advanced strategy...')

    # * Remove sentinel values (-9999) and replace with NaN
    df = df.replace({-9999: np.nan, '-9999': np.nan})

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    core_num    = [c for c in CORE_ATTRIBUTES       if c in num_cols]
    support_num = [c for c in SUPPORTING_ATTRIBUTES if c in num_cols]
    support_cat = [c for c in SUPPORTING_ATTRIBUTES if c in cat_cols]
    default_num = [c for c in num_cols if c not in core_num + support_num]
    default_cat = [c for c in cat_cols if c not in support_cat]

    print(f"\n    Core Numeric columns ({len(core_num)}): {core_num}")
    print(f"\n    Support Numeric columns ({len(support_num)}): {support_num}")
    print(f"\n    Support Categorical columns ({len(support_cat)}): {support_cat}")
    print(f"\n    Default Numeric columns ({len(default_num)}): {default_num}")
    print(f"\n    Default Categorical columns ({len(default_cat)}): {default_cat}")

    core_imp        = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
    support_num_imp = KNNImputer(n_neighbors=5)
    support_cat_imp = SimpleImputer(strategy='most_frequent')
    default_num_imp = SimpleImputer(strategy='median')
    default_cat_imp = SimpleImputer(strategy='most_frequent')

    transformer = ColumnTransformer([
        ('core_num',    core_imp,        core_num),
        ('sup_num',     support_num_imp, support_num),
        ('sup_cat',     support_cat_imp, support_cat),
        ('num_def',     default_num_imp, default_num),
        ('cat_def',     default_cat_imp, default_cat),
    ], remainder='drop')

    imputed_values = transformer.fit_transform(df)
    out_cols = core_num + support_num + support_cat + default_num + default_cat
    print(f"\n    Imputed shape: {imputed_values.shape}")

    df_imputed = pd.DataFrame(imputed_values, columns=out_cols, index=df.index)
    print(f"\n    Imputed DataFrame columns ({len(df_imputed.columns)}): {df_imputed.columns.tolist()}")

    missing_after = df_imputed.isnull().sum().sum()
    print(f"\nTotal missing after imputation: {missing_after}")

    return df_imputed

def main():
    path = '../data/asd_data/Phenotypic_V1_0b_preprocessed1_from_shawon.csv'
    df = load_data(path)

    missing_values = missing_value_report(df)
    plot_missing_distribution(missing_values)
    
    thresholds = {'core': 0.8, 'support': 0.5, 'rare': 0.3, 'default': 0.5}
    df_clean, dropped = drop_by_missing_threshold(missing_values, df, thresholds=thresholds)
    remaining_missing_values = missing_value_report(df_clean)
    plot_missing_distribution(remaining_missing_values)
    remaining_missing_values.to_csv('../output/remaining_missing_values.csv')

    df_imputed = impute_missing(df_clean)
    df_imputed.to_csv('../output/imputed_data.csv', index=False)

if __name__ == '__main__':
    main()
