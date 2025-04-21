import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """
    Load phenotypic data from the ABIDE dataset into a pandas DataFrame.

    Motivation:
    Researchers need a standardized way to ingest raw phenotypic CSV files before any preprocessing.
    This function centralizes file reading and ensures consistent delimiter usage.

    :param path: str
        File system path to the CSV file containing ABIDE phenotypic data.
    :return: pandas.DataFrame
        The raw phenotypic DataFrame for subsequent cleaning and analysis.
    """
    df = pd.read_csv(path, sep=',')
    return df

def missing_value_report(df):
    """
    Generate a summary of missing values for each column in the DataFrame.

    Motivation:
    Understanding the extent and pattern of missing data guides decisions on column removal
    or value imputation, which is critical to avoid biased or invalid model inputs.

    :param df: pandas.DataFrame
        DataFrame to analyze for missing values.
    :return: pandas.DataFrame
        A two-column DataFrame where 'Total Missing' is the count of NaNs
        and '% Missing' is the percentage relative to the DataFrame length.
    """
    total = df.isnull().sum()
    percent = 100 * total / len(df)
    report = pd.concat([total, percent], axis=1, keys=['Total Missing', '% Missing'])
    return report

def drop_high_missing(df, threshold=0.5):
    """
    Remove columns exceeding a specified threshold of missing values.

    Motivation:
    Columns with excessive missingness (e.g., >50%) often provide little usable information
    and can introduce noise. Dropping them streamlines downstream processing.

    :param df: pandas.DataFrame
        Input DataFrame potentially containing many columns with missing data.
    :param threshold: float, default=0.5
        Proportion of allowed missing values before dropping (0 < threshold < 1).
    :return: tuple
        - pandas.DataFrame: DataFrame with high-missing columns removed.
        - list[str]: Names of the dropped columns for record-keeping.
    """
    missing_report = missing_value_report(df)
    cols_to_drop = missing_report[missing_report['% Missing'] > 100 * threshold].index.tolist()
    df_clean = df.drop(columns=cols_to_drop)
    return df_clean, cols_to_drop

def impute_missing(df):
    """
    Fill missing values using median for numeric columns and mode for categorical ones.

    Motivation:
    Imputation prevents data loss when rows contain NaNs. Median is robust to outliers
    for numerical features, while the most frequent category preserves data distribution
    in categorical features.

    :param df: pandas.DataFrame
        Cleaned DataFrame after dropping high-missing columns.
    :return: pandas.DataFrame
        DataFrame where missing values have been replaced.
    """
    df_num = df.select_dtypes(include=[np.number])
    df_cat = df.select_dtypes(include=['object', 'category'])

    imp_num = SimpleImputer(strategy='median')
    imp_cat = SimpleImputer(strategy='most_frequent')

    df_num_imp = pd.DataFrame(imp_num.fit_transform(df_num), columns=df_num.columns)
    df_cat_imp = pd.DataFrame(imp_cat.fit_transform(df_cat), columns=df_cat.columns)

    df_imputed = pd.concat([df_num_imp, df_cat_imp], axis=1)
    return df_imputed

def compute_correlation(df, method='pearson'):
    """
    Calculate the pairwise correlation matrix for numeric variables.

    Motivation:
    Correlation analysis helps detect multicollinearity, identify redundant features,
    and explore relationships between predictors and the outcome (DX_GROUP).
    Pearson measures linear relationships, while Spearman can capture monotonic trends.

    :param df: pandas.DataFrame
        Imputed DataFrame including numeric predictors and the target column.
    :param method: str, default='pearson'
        Correlation method: 'pearson' for linear, 'spearman' for rank-based.
    :return: pandas.DataFrame
        Square correlation matrix of all numeric variables.
    """
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr(method=method)
    return corr

def select_features_by_target_corr(corr_matrix, target_col, threshold=0.1):
    """
    Identify features with absolute correlation above a threshold relative to the target.

    Motivation:
    Selecting predictors most associated with the diagnostic label (DX_GROUP)
    helps focus the model on variables with predictive signal and reduces dimensionality.

    :param corr_matrix: pandas.DataFrame
        Correlation matrix from compute_correlation().
    :param target_col: str
        Name of the target variable column (e.g., 'DX_GROUP').
    :param threshold: float, default=0.1
        Minimum absolute correlation value to consider a feature relevant.
    :return: list[str]
        List of feature names with |correlation| > threshold, excluding the target.

    Note on correlating with DX_GROUP:
    DX_GROUP encodes diagnosis (1=ASD, 2=Control). Computing its correlation
    with predictors quantifies how strongly each feature varies across diagnostic groups.
    A higher absolute correlation suggests that feature distinguishes ASD vs controls,
    guiding feature selection and interpretability.
    """
    if target_col not in corr_matrix.columns:
        raise ValueError(f"{target_col} not in correlation matrix")
    corr_with_target = corr_matrix[target_col].abs().sort_values(ascending=False)
    selected = corr_with_target[corr_with_target > threshold].index.tolist()
    selected.remove(target_col)
    return selected

def run_pca(df, n_components=0.95):
    """
    Perform Principal Component Analysis on standardized numeric data.

    Motivation:
    PCA reduces dimensionality by transforming features into orthogonal components
    that capture the greatest variance. Specifying n_components as a float
    retains enough PCs to explain that fraction of total variance (e.g., 0.95).

    :param df: pandas.DataFrame
        Imputed DataFrame with only numeric predictor columns (target dropped).
    :param n_components: float or int, default=0.95
        If float, the fraction of variance to retain; if int, the number of components.
    :return: tuple
        - sklearn.decomposition.PCA: fitted PCA object (with explained_variance_)
        - pandas.DataFrame: transformed data in principal component space.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=columns)
    return pca, df_pca

def plot_correlation_matrix(corr, figsize=(12,10)):
    """
    Visualize the correlation matrix as a heatmap.

    Motivation:
    A heatmap provides an intuitive overview of feature interrelationships,
    highlighting clusters of highly correlated variables and potential multicollinearity.

    :param corr: pandas.DataFrame
        Square correlation matrix.
    :param figsize: tuple, default=(12,10)
        Figure size for the plot.
    """
    plt.figure(figsize=figsize)
    plt.imshow(corr, aspect='auto', cmap='viridis')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label='Correlation coefficient')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def list_top_correlations(corr_matrix, target_col, n=10):
    """
    Return the top-N features most correlated with the target.

    Motivation:
    Quickly surface the strongest associations between predictors and diagnosis,
    aiding in feature prioritization and biological interpretation.

    :param corr_matrix: pandas.DataFrame
        Correlation matrix containing target and predictors.
    :param target_col: str
        Name of the target variable column.
    :param n: int, default=10
        Number of top features to return.
    :return: pandas.Series
        Sorted absolute correlations of top-N predictors with the target.
    """
    corr_with_target = corr_matrix[target_col].abs().drop(target_col)
    return corr_with_target.sort_values(ascending=False).head(n)

def split_data(df, features, target_col='DX_GROUP', test_size=0.2, random_state=42):
    """
    Split DataFrame into train/test arrays for features and target.

    :param df: pandas.DataFrame
    :param features: list of column names to use as predictors
    :param target_col: name of the target column ('DX_GROUP')
    :param test_size: proportion of data to reserve for test
    :param random_state: seed for reproducibility
    :return: X_train, X_test, y_train, y_test
    """
    X = df[features]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def check_class_balance(y):
    """
    Print the distribution of classes in the target vector.

    :param y: pandas.Series (target)
    """
    counts = y.value_counts()
    pct = 100 * counts / counts.sum()
    print('Class distribution:')
    print(pd.concat([counts, pct], axis=1, keys=['Count','Percent']))

def build_pipeline(model, use_scaling=True):
    """
    Create a sklearn Pipeline with optional scaling and the given estimator.

    :param model: an instantiated sklearn estimator (e.g., RandomForestClassifier())
    :param use_scaling: whether to include a StandardScaler step
    :return: a sklearn Pipeline
    """
    steps = []
    if use_scaling:
        steps.append(('scaler', StandardScaler()))
    steps.append(('clf', model))
    return Pipeline(steps)

def evaluate_model(pipeline, X_test, y_test):
    """
    Compute and print common classification metrics.

    :param pipeline: trained sklearn Pipeline
    :param X_test: test features
    :param y_test: true test labels
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline, 'predict_proba') else None

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, pos_label=1))
    print('Recall:', recall_score(y_test, y_pred, pos_label=1))
    print('F1-score:', f1_score(y_test, y_pred, pos_label=1))
    if y_prob is not None:
        print('ROC AUC:', roc_auc_score(y_test, y_prob))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))



def main():
    path = '../data/asd_data/Phenotypic_V1_0b_preprocessed1.csv'
    df = load_data(path)

    # * 1. Missing data analysis
    report = missing_value_report(df)
    print('Missing Values Report:')
    print(report)

    # * 2. Drop columns and impute
    df_clean, dropped = drop_high_missing(df, threshold=0.5)
    print(f'Dropped columns (>{50}% missing): {dropped}')
    df_imputed = impute_missing(df_clean)

    # * 3. Correlation analysis
    corr_matrix = compute_correlation(df_imputed, method='pearson')
    plot_correlation_matrix(corr_matrix)
    top_feats = list_top_correlations(corr_matrix, target_col='DX_GROUP', n=10)
    print('Top correlated features with DX_GROUP:')
    print(top_feats)

    # * 4. Feature selection & dimensionality reduction
    selected_features = select_features_by_target_corr(corr_matrix, target_col='DX_GROUP', threshold=0.1)
    print(f'Selected features by correlation threshold: {selected_features}')
    pca, df_pca = run_pca(df_imputed[selected_features], n_components=0.95)
    print(f'Number of PCA components to explain 95% variance: {df_pca.shape[1]}')

    # Test
    selected_feats = ['func_mean_fd', 'func_num_fd', 'func_perc_fd', 'func_quality', 'func_outlier']
    X_train, X_test, y_train, y_test = split_data(df_imputed, selected_feats)
    check_class_balance(y_train)
    
    rf_pipeline = build_pipeline(RandomForestClassifier(random_state=42))
    print(rf_pipeline)

    rf_pipeline.fit(X_train, y_train)
    evaluate_model(rf_pipeline, X_test, y_test)

if __name__ == '__main__':
    main()
