import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(path):
    """
    Load phenotypic data from ABIDE dataset
    :param path: path to the CSV file
    :return: pandas DataFrame
    """
    df = pd.read_csv(path, sep=',')
    return df

def missing_value_report(df):
    """
    Report missing values by column in a DataFrame.
    :param df: DataFrame to analyze
    :return: DataFrame with total and percentage of missing values for each column
    """
    total = df.isnull().sum()
    percent = 100 * total / len(df)
    report = pd.concat([total, percent], axis=1, keys=['Total Missing', '% Missing'])
    return report

def drop_high_missing(df, threshold=0.5):
    """
    Remove columns with more than (threshold * 100%) missing values.
    :param df: DataFrame to filter
    :param threshold: proportion of missing values for removal (default 0.5)
    :return: DataFrame with columns dropped and list of dropped columns
    """
    missing_report = missing_value_report(df)
    cols_to_drop = missing_report[missing_report['% Missing'] > 100*threshold].index.tolist()
    df_clean = df.drop(columns=cols_to_drop)
    return df_clean, cols_to_drop

def impute_missing(df):
    """
    Input missing values: median for numeric and mode for categorical.
    :param df: DataFrame to impute
    :return: DataFrame with missing values imputed
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
    Compute correlation matrix for numeric variables.
    :param df: DataFrame to analyze
    :param method: correlation method ('pearson', 'spearman')
    :return: DataFrame with correlation matrix
    """
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr(method=method)
    return corr

def select_features_by_target_corr(corr_matrix, target_col, threshold=0.1):
    """
    Select features whose absolute correlation with the target column exceeds a threshold.
    :param corr_matrix: correlation DataFrame
    :param target_col: name of the target column
    :param threshold: minimum correlation value
    :return: list of selected features
    """
    if target_col not in corr_matrix.columns:
        raise ValueError(f"{target_col} not in correlation matrix")
    corr_with_target = corr_matrix[target_col].abs().sort_values(ascending=False)
    selected = corr_with_target[corr_with_target > threshold].index.tolist()
    selected.remove(target_col)
    return selected

def run_pca(df, n_components=0.95):
    """
    Execute PCA over normalized data and return components that explain variance.
    :param df: imputed numeric DataFrame
    :param n_components: float (cumulative var.) or int (number of components)
    :return: PCA object, DataFrame of principal components
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=columns)
    return pca, df_pca

if __name__ == '__main__':
    path = "../data/asd_data/Phenotypic_V1_0b_preprocessed1.csv"
    df = load_data(path)

    print("Missing Values Report:")
    report = missing_value_report(df)
    print(report)

    print(f"DataFrame shape before dropping: {df.shape}")
    df_clean, dropped = drop_high_missing(df, threshold=0.5)
    print(f"Dropped columns (>{50}% missing): {dropped}")
    print(f"DataFrame shape after dropping: {df_clean.shape}")

    df_imputed = impute_missing(df_clean)

    corr_matrix = compute_correlation(df_imputed, method='pearson')
    print(f"Correlation matrix shape: {corr_matrix.shape}")

    features = select_features_by_target_corr(corr_matrix, target_col='DX_GROUP', threshold=0.1)
    print(f"Selected features by correlation with DX_GROUP (>0.1): {features}")

    pca, df_pca = run_pca(df_imputed.drop(columns=['DX_GROUP']), n_components=0.95)
    print(f"Number of PCA components to explain 95% var: {df_pca.shape[1]}")
