import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def load_data(path):
    print(f"\n[1] Loading data from: {path}")
    df = pd.read_csv(path, sep=',')
    print(f"    Data shape: {df.shape}")
    print(df.columns.tolist())
    return df

def missing_value_report(df):
    report = df.isnull().sum().rename("Total Missing").to_frame()
    report['% Missing'] = 100 * report['Total Missing'] / len(df)
    report = report.sort_values('% Missing', ascending=False)
    report.drop("Total Missing", axis=1, inplace=True)
    print('\n[2] Missing value summary:')
    print(report.head(106))
    return report

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
    thresholds = {
      'core': 0.7,
      'support': 0.5,
      'rare': 0.3,
      'default': 0.5
    }
    and whitelist of core features.
    Returns cleaned df and list of dropped columns.
    """
    # Define feature categories
    core = [
        # Label e classificação diagnóstica
        'DX_GROUP',       # alvo
        'DSM_IV_TR',      # diagnóstico segundo DSM-IV

        # ADI-R: Entrevista Diagnóstica
        'ADI_R_SOCIAL_TOTAL_A',    # Social (63% missing)
        'ADI_R_VERBAL_TOTAL_BV',   # Verbal (63% missing)
        'ADI_RRB_TOTAL_C',         # Repetitive behaviors (63% missing)
        'ADI_R_ONSET_TOTAL_D',     # Onset (70% missing)
        'ADI_R_RSRCH_RELIABLE',    # Confiabilidade (63% missing)
        

        # ADOS: Observação Diagnóstica
        'ADOS_MODULE',               # Módulo aplicado (52% missing)
        'ADOS_TOTAL',                # Total (60% missing)
        'ADOS_COMM',                 # Comunicação (62.5% missing)
        'ADOS_SOCIAL',               # Social (62.5% missing)
        'ADOS_STEREO_BEHAV',         # Stereotipias (66% missing)
        'ADOS_RSRCH_RELIABLE',       # Confiabilidade (66% missing)
        
        # SRS e triagem de espectro
        'SRS_RAW_TOTAL',   # Escore total SRS (63% missing)
    ]
    
    support = [
        # Demografia & batch
        'AGE_AT_SCAN', 
        'SEX', 
        'EYE_STATUS_AT_SCAN',
        'HANDEDNESS_CATEGORY',
        'SITE_ID',         # controle de site
        'SUB_IN_SMP',      # flag de inclusão
        
        # QI
        'FIQ',             # Full IQ (3% missing)
        'VIQ',             # Verbal IQ (16%)
        'PIQ',             # Performance IQ (14%)
        'FIQ_TEST_TYPE',   # Tipo de teste (15%)
        'VIQ_TEST_TYPE', 
        'PIQ_TEST_TYPE',
        
        # Status clínico
        'CURRENT_MED_STATUS',  # 26% missing
        
        # Qualidade anatômica
        'anat_cnr','anat_efc','anat_fber','anat_fwhm','anat_qi1','anat_snr',
        
        # Qualidade funcional & movimento
        'func_efc','func_fber','func_fwhm','func_dvars','func_outlier','func_quality',
        'func_mean_fd','func_num_fd','func_perc_fd','func_gsr',
        
        # QC manual (sem notas textuais)
        'qc_rater_1','qc_anat_rater_2','qc_func_rater_2','qc_anat_rater_3','qc_func_rater_3'
    ]
    
    rare = [
        # Triagem secundária
        'SCQ_TOTAL',    # 87% missing
        'AQ_TOTAL',     # 94% missing

        # Subescalas detalhadas do SRS
        'SRS_AWARENESS','SRS_COGNITION','SRS_COMMUNICATION','SRS_MOTIVATION','SRS_MANNERISMS', # 94%

        # Vineland (funcionamento adaptativo)
        *[c for c in df.columns if c.startswith('VINELAND')],  # 83%

        # WISC-IV subtests
        *[c for c in df.columns if c.startswith('WISC_IV')],   # 95%

        # Notas de QC (textuais) e outros
        'qc_notes_rater_1','qc_anat_notes_rater_2','qc_func_notes_rater_2',
        'qc_anat_notes_rater_3','qc_func_notes_rater_3',
        'MEDICATION_NAME','COMORBIDITY','OFF_STIMULANTS_AT_SCAN'
    ]

    to_drop = []
    for col, pct in report['% Missing'].items():
        if col in core:
            limit = thresholds['core']
        elif col in support:
            limit = thresholds['support']
        elif col in rare:
            limit = thresholds['rare']
        else:
            limit = thresholds['default']
        if pct/100 > limit:
            to_drop.append(col)

    print(f"\n[3] Conditionally dropping {len(to_drop)} columns based on thresholds: {to_drop}")
    df_clean = df.drop(columns=to_drop)
    print(f"\nNew shape after conditional drop: {df_clean.shape}")
    print(f"[3] Remaining columns: {df_clean.columns.tolist()}")
    return df_clean, to_drop
    
def impute_missing(df):
    print('\n[4] Imputing missing values...')
    print(f"Num columns: {df.select_dtypes(include=[np.number])}")
    print(f"Cat columns: {df.select_dtypes(include=['object', 'category'])}")
    df_num = df.select_dtypes(include=[np.number])
    df_cat = df.select_dtypes(include=['object', 'category'])
    imp_num = SimpleImputer(strategy='median')
    imp_cat = SimpleImputer(strategy='most_frequent')
    df_num_imp = pd.DataFrame(imp_num.fit_transform(df_num), columns=df_num.columns)
    df_cat_imp = pd.DataFrame(imp_cat.fit_transform(df_cat), columns=df_cat.columns)
    df_imputed = pd.concat([df_num_imp, df_cat_imp], axis=1)
    print(f"    Imputed data shape: {df_imputed.shape}")
    return df_imputed

def compute_correlation(df, method='pearson'):
    print(f"[5] Computing {method} correlation matrix...")
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr(method=method)
    print(f"    Correlation matrix shape: {corr.shape}")
    return corr

def select_features_by_target_corr(corr_matrix, target_col, threshold=0.1):
    if target_col not in corr_matrix.columns:
        raise ValueError(f"{target_col} not in correlation matrix")
    vals = corr_matrix[target_col].abs().sort_values(ascending=False)
    selected = vals[vals > threshold].index.tolist()
    selected.remove(target_col)
    print(f"[6] Features selected (|corr|>{threshold}): {selected}")
    return selected

def run_pca(df, n_components=0.95):
    print(f"[7] Running PCA (n_components={n_components})...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=cols)
    print(f"    PCA result shape: {df_pca.shape}")
    return pca, df_pca

def split_data(df, features, target_col='DX_GROUP', test_size=0.2, random_state=42):
    print(f"[8] Splitting data (test_size={test_size})...")
    X = df[features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"    Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def check_class_balance(y):
    counts = y.value_counts()
    pct = 100 * counts / counts.sum()
    df_balance = pd.concat([counts, pct], axis=1, keys=['Count','Percent'])
    print(f"[9] Class balance:\n{df_balance}")
    return df_balance

def build_pipeline(model, use_scaling=True):
    steps = []
    if use_scaling:
        steps.append(('scaler', StandardScaler()))
    steps.append(('clf', model))
    pipeline = Pipeline(steps)
    print(f"[10] Built pipeline: {pipeline}")
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    print(f"[11] Evaluating model: {pipeline.named_steps['clf']}...")
    # assume pipeline already fitted
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline, 'predict_proba') else None
    print("    Metrics:")
    print(f"      Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"      Precision: {precision_score(y_test, y_pred, pos_label=1):.4f}")
    print(f"      Recall   : {recall_score(y_test, y_pred, pos_label=1):.4f}")
    print(f"      F1-score : {f1_score(y_test, y_pred, pos_label=1):.4f}")
    if y_prob is not None:
        print(f"      ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("    Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("    Classification Report:")
    print(classification_report(y_test, y_pred))

def compare_pipelines(pipelines, X_train, X_test, y_train, y_test):
    print(f"[12] Comparing {len(pipelines)} pipelines...")
    results = []
    for name, pipe in pipelines.items():
        print(f"    Fitting pipeline: {name}")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        auc = None
        if hasattr(pipe, 'predict_proba'):
            auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC_AUC': auc
        })
    df_results = pd.DataFrame(results)
    print("    Comparison results:")
    print(df_results)
    return df_results


def main():
    path = '../data/asd_data/Phenotypic_V1_0b_preprocessed1_from_shawon.csv'
    df = load_data(path)

    missing_values = missing_value_report(df)
    plot_missing_distribution(missing_values)
    
    thresholds = {'core': 0.8, 'support': 0.5, 'rare': 0.3, 'default': 0.5}
    df_clean, dropped = drop_by_missing_threshold(missing_values, df, thresholds=thresholds)
    # plot_missing_distribution(df_clean)
    remaining_missing_values = missing_value_report(df_clean)
    plot_missing_distribution(remaining_missing_values)
    remaining_missing_values.to_csv('remaining_missing_values.csv')




    # # Step 4: Impute Missing Values
    # df_imputed = impute_missing(df_clean)

    # # Step 5: Correlation Analysis
    # corr_matrix = compute_correlation(df_imputed, method='pearson')

    # # Step 6: Feature Selection by Correlation
    # target = 'DX_GROUP'
    # selected_features = select_features_by_target_corr(corr_matrix, target_col=target, threshold=0.1)

    # # Step 7: Dimensionality Reduction (PCA)
    # pca_model, df_pca = run_pca(df_imputed[selected_features], n_components=0.95)

    # # Step 8: Train/Test Split
    # X_train, X_test, y_train, y_test = split_data(df_imputed, selected_features, target_col=target, test_size=0.2)
    # _ = check_class_balance(y_train)

    # # Step 9: Build and Train Model
    # rf_pipeline = build_pipeline(RandomForestClassifier(random_state=42))
    # rf_pipeline.fit(X_train, y_train)

    # # Step 10: Evaluate Single Model
    # evaluate_model(rf_pipeline, X_test, y_test)

    # # Step 11: Compare Multiple Pipelines
    # pipelines = {
    #     'RandomForest': rf_pipeline,
    #     'LogisticRegression': build_pipeline(LogisticRegression(max_iter=1000))
    # }
    # compare_pipelines(pipelines, X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()

    # Entender melhor os dados - remover colunas que não são relevantes
    
    # Classification models to try:
    # Random Forest
    # Logistic Regression
    # SVM
    # KNN
    # DT
    # ExtraTrees
    # XGBoost
    # LightGBM
    # CatBoost
    # AdaBoost
    # Naive Bayes
    # MLP
    # Stacking / Voting Ensembles
    
    # Em vez de CV, usar StratifiedKFold para medir variabilidade?
    # Comparar com a literatura e utilizar o que estao usando.
    
    # Hyperparameter tuning:
    # GridSearchCV, RandomizedSearchCV, Optuna, etc.
    
    # Pipeline unico:
    # Imputer -> Scaler -> Feature Selection -> PCA -> Estimator
    
    # Verificar se tem desbalanceamento de classes e aplicar oversampling (SMOTE), etc

    # Métricas:
    # AUCROC, F1, Precision, Recall, Confusion Matrix, Classification Report, Precision-Recall Curve
    
    # Plot de learning curves para ver se tem overfitting ou underfitting
    
    # Comparar PCA com Feature Selection
    
    # Gerar Relatorio Automatizado com pandas e gráficos de comparação
    