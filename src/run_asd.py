import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Carregar dados
def load_data(path):
    """
    Carrega o CSV de dados fenotípicos do ABIDE.
    :param path: caminho para o arquivo CSV
    :return: DataFrame pandas
    """
    df = pd.read_csv(path, sep=',')
    return df

# 2. Identificar e reportar valores ausentes
def missing_value_report(df):
    """
    Relatório de valores ausentes por coluna.
    :param df: DataFrame
    :return: Series com total e percentuais de missing
    """
    total = df.isnull().sum()
    percent = 100 * total / len(df)
    report = pd.concat([total, percent], axis=1, keys=['Total Missing', '% Missing'])
    return report

# 3. Remover colunas com alta proporção de missing
def drop_high_missing(df, threshold=0.5):
    """
    Remove colunas com mais de threshold*100% de valores ausentes.
    :param df: DataFrame
    :param threshold: proporção de missing para remoção (default 0.5)
    :return: DataFrame filtrado
    """
    missing_report = missing_value_report(df)
    cols_to_drop = missing_report[missing_report['% Missing'] > 100*threshold].index.tolist()
    df_clean = df.drop(columns=cols_to_drop)
    return df_clean, cols_to_drop

# 4. Imputação de valores ausentes
def impute_missing(df):
    """
    Imputa valores ausentes: median para numéricos e moda para categóricos.
    :param df: DataFrame
    :return: DataFrame imputado
    """
    df_num = df.select_dtypes(include=[np.number])
    df_cat = df.select_dtypes(include=['object', 'category'])

    # Imputadores
    imp_num = SimpleImputer(strategy='median')
    imp_cat = SimpleImputer(strategy='most_frequent')

    df_num_imp = pd.DataFrame(imp_num.fit_transform(df_num), columns=df_num.columns)
    df_cat_imp = pd.DataFrame(imp_cat.fit_transform(df_cat), columns=df_cat.columns)

    # Combinar
    df_imputed = pd.concat([df_num_imp, df_cat_imp], axis=1)
    return df_imputed

# 5. Análise de correlação
def compute_correlation(df, method='pearson'):
    """
    Computa matriz de correlação para variáveis numéricas.
    :param df: DataFrame
    :param method: método de correlação ('pearson', 'spearman')
    :return: matriz de correlação
    """
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr(method=method)
    return corr

# 6. Seleção de features por correlação com o alvo
def select_features_by_target_corr(corr_matrix, target_col, threshold=0.1):
    """
    Seleciona features cuja correlação absoluta com a coluna-alvo excede um threshold.
    :param corr_matrix: DataFrame de correlação
    :param target_col: nome da coluna-alvo
    :param threshold: valor mínimo de correlação
    :return: lista de features selecionadas
    """
    if target_col not in corr_matrix.columns:
        raise ValueError(f"{target_col} not in correlation matrix")
    corr_with_target = corr_matrix[target_col].abs().sort_values(ascending=False)
    selected = corr_with_target[corr_with_target > threshold].index.tolist()
    selected.remove(target_col)
    return selected

# 7. PCA para análise de variância explicada
def run_pca(df, n_components=0.95):
    """
    Executa PCA sobre dados normalizados e retorna componentes que explicam variância.
    :param df: DataFrame numérico imputado
    :param n_components: float (var. acumulada) ou int (número de componentes)
    :return: PCA object, DataFrame das componentes principais
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=columns)
    return pca, df_pca

if __name__ == '__main__':
    # Exemplo de uso
    path = 'data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv'
    df = load_data(path)

    print("Missing Values Report:")
    report = missing_value_report(df)
    print(report)

    df_clean, dropped = drop_high_missing(df, threshold=0.5)
    print(f"Dropped columns (>{50}% missing): {dropped}")

    df_imputed = impute_missing(df_clean)

    corr_matrix = compute_correlation(df_imputed, method='pearson')
    print("Correlation matrix computed")

    features = select_features_by_target_corr(corr_matrix, target_col='DX_GROUP', threshold=0.1)
    print(f"Selected features by correlation with DX_GROUP (>0.1): {features}")

    pca, df_pca = run_pca(df_imputed.drop(columns=['DX_GROUP']), n_components=0.95)
    print(f"Number of PCA components to explain 95% var: {df_pca.shape[1]}")
