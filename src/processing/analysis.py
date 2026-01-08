# * Used for Pearson, Spearman and PPS (Predictive Power Score) calculations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ppscore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ppscore as pps
import seaborn as sns

from src.shared import config, utils

# Get logger instance for this module
logger = logging.getLogger(__name__)

# 2. Define Grouping Logic (Regex-like mapping)
# Dictionary: { 'Substring to find': 'Readable Group Name' }
# Order matters! More specific patterns should come first if needed.
GROUP_PATTERNS = {
    # Families groups
    'WISC_IV': 'Bateria WISC-IV (QI)',
    'WISC_': 'Outros Escores WISC (QI)',
    'VINELAND': 'Escalas Vineland (Comportamento Adaptativo)',
    'SRS_': 'Sub-escalas SRS (Responsividade Social)',
    'ADI_R_': 'Sub-escores ADI-R',
    'ADI_RRB': 'Comportamento Repetitivo ADI-R',
    'ADOS_': 'Sub-escores ADOS',
    'SCQ_': 'Questionário de Comunicação Social (SCQ)',

    # Quality Control (QC) Metrics
    'qc_.*notes': 'Anotações de Controle de Qualidade (Texto)',
    'qc_.*rater': 'Avaliadores de Qualidade (QC Rater)',
    'anat_': 'Métricas Anatômicas (MRI)',
    'func_': 'Métricas Funcionais (fMRI)',

    # Individuals (Direct Translation)
    'AQ_TOTAL': 'Quociente de Autismo Total (AQ)',
    'COMORBIDITY': 'Indicadores de Comorbidade',
    'AGE_AT_MPRAGE': 'Idade na Ressonância',
    'OFF_STIMULANTS_AT_SCAN': 'Sem Estimulantes no Scan',
    'MEDICATION_NAME': 'Nome da Medicação',
    'BMI': 'Índice de Massa Corporal (IMC)',
    'HANDEDNESS_SCORES': 'Escores de Lateralidade',
    'HANDEDNESS_CATEGORY': 'Categoria de Lateralidade',
    'CURRENT_MED_STATUS': 'Status de Medicação Atual',
    'VIQ_TEST_TYPE': 'Tipo de Teste QI Verbal',
    'PIQ_TEST_TYPE': 'Tipo de Teste QI Performance',
    'VIQ': 'QI Verbal',
    'PIQ': 'QI de Performance',
    'FIQ': 'QI Total'
}

def plot_missing_distribution(report_df: pd.DataFrame, fileName: str):
    """
    Generates a horizontal bar plot of missing values.
    Intelligent grouping: Aggregates related features (e.g., 'WISC_IV_...') 
    into families to reduce visual clutter and improve readability.
    """
    logger.info("Generating Top 20 missing values plot...")
    utils.apply_plot_style()

    if report_df.empty:
        logger.warning("No missing values to plot.\n")
        return

    col_name = config.PERCENT_MISSING_COLUMN
    if col_name in report_df.columns:
        df_plot = report_df[[col_name]].copy()
    else:
        df_plot = report_df.iloc[:, 0].to_frame()

    df_plot.columns = ['pct']
    df_plot = df_plot[df_plot['pct'] > 0].copy()
    df_plot['feature_group'] = df_plot.index.map(assign_group)
    df_grouped = df_plot.groupby('feature_group')['pct'].mean().sort_values(ascending=False)
    top_missing = df_grouped.head(20)

    if top_missing.empty:
        logger.warning("No columns with missing values found to plot.\n")
        return

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=top_missing.values,
        y=top_missing.index,
        color='#4A90E2',
        edgecolor='none'
    )
    plt.title('Top 20 Feature Groups with Missing Data', pad=20)
    plt.xlabel('Percentage of Missing Values (%)')
    plt.ylabel('') # Feature names are self-explanatory
    plt.xlim(0, 100)
    plt.grid(axis='x', alpha=0.3)

    for i, v in enumerate(top_missing.values):
        plt.text(v + 1, i + 0.25, f"{v:.1f}%", color='black', fontsize=10)

    # Remove top and right spines for cleaner look
    sns.despine()
    plt.tight_layout()

    output_path = config.OUTPUT_DIR / fileName
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Plot saved to: {output_path}\n")

def assign_group(feature_name):
    for pattern, group_name in GROUP_PATTERNS.items():
        if pattern in feature_name:
            return group_name
        # Handle Regex cases
        if '.*' in pattern: 
            import re
            if re.search(pattern, feature_name):
                return group_name
    return feature_name # If no match, keep original name

def plot_class_balance(df: pd.DataFrame, target_col: str):
    """
    Plots the distribution of the target variable (Autism vs Control).
    Standardized for academic publication.
    """
    logger.info("Generating Class Balance Plot...")
    utils.apply_plot_style()

    # Check if target exists
    if target_col not in df.columns:
        logger.warning(f"Target column {target_col} not found for balance plot.")
        return

    # Count values
    counts = df[target_col].value_counts(normalize=True) * 100
    counts_raw = df[target_col].value_counts()

    # Map labels if possible (assuming 1=Autism, 2=Control based on Config)
    # We create a display mapping for the plot
    labels_map = {1: 'Autismo (TEA)', 2: 'Controle (CT)'}
    if 0 in counts.index: # If already mapped to 0/1
        labels_map = {1: 'Autismo (TEA)', 0: 'Controle (CT)'}

    plot_data = pd.DataFrame({
        'Label': [labels_map.get(x, str(x)) for x in counts.index],
        'Percentage': counts.values,
        'Count': counts_raw.values,
        'Original': counts.index
    })

    plt.figure(figsize=(8, 6))
    bars = sns.barplot(
        data=plot_data,
        x='Label',
        y='Percentage',
        edgecolor='black',
        hue='Original',
        legend=False,
        palette=['#E74C3C', '#4A90E2'] # Red for Autism, Blue for Control
    )
    for i, row in plot_data.iterrows():
        # Text: "52.3% (N=539)"
        text = f"{row['Percentage']:.1f}%\n(N={row['Count']})"
        bars.text(i, row['Percentage'] + 1, text, ha='center', color='black', fontsize=12)

    plt.title('Distribuição das Classes (Diagnóstico)', pad=20)
    plt.ylabel('Porcentagem do Dataset (%)')
    plt.xlabel('')
    plt.ylim(0, 100) # Fix scale usually helps context

    sns.despine()
    plt.tight_layout()

    output_path = config.OUTPUT_DIR / "02_class_balance.png"
    plt.savefig(output_path)
    logger.info(f"Plot saved to: {output_path}\n")

def plot_stratified_missingness(df: pd.DataFrame, target_col: str, features_to_check: list):
    """
    Plots missingness percentage separate by Target Class.
    This proves if missing data is correlated with the Class (MNAR).
    """
    logger.info("Generating Stratified Missingness Plot...")
    utils.apply_plot_style()
    
    if target_col not in df.columns:
        return

    # Prepare data storage
    plot_data = []

    labels_map = {1: 'Autismo (TEA)', 2: 'Controle (CT)'}
    if set(df[target_col].unique()) == {0, 1}:
        labels_map = {1: 'Autismo (TEA)', 0: 'Controle (CT)'}

    for feat in features_to_check:
        if feat not in df.columns:
            continue

        # Group by target and calculate missing %
        # We calculate: count of nulls / count of rows * 100
        missing_by_class = df.groupby(target_col)[feat].apply(lambda x: x.isnull().mean() * 100)

        for class_id, pct in missing_by_class.items():
            plot_data.append({
                'Feature': feat,
                'Class': labels_map.get(class_id, str(class_id)),
                'Missing %': pct
            })

    if not plot_data:
        logger.warning("No features found for stratified missingness.")
        return

    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_plot,
        x='Feature',
        y='Missing %',
        edgecolor='black',
        hue='Class',
        palette=['#4A90E2', '#E74C3C'] # Blue vs Red
    )
    plt.title('Dados Ausentes Estratificados por Diagnóstico', pad=20)
    plt.ylabel('Porcentagem de Ausência (%)')
    plt.xlabel('')
    plt.ylim(0, 105)
    plt.legend(title='Grupo')
    plt.grid(axis='y', alpha=0.3)
    # Rotate x labels if needed
    plt.xticks(rotation=45, ha='right')

    sns.despine()
    plt.tight_layout()

    output_path = config.OUTPUT_DIR / "02_missingness_stratified.png"
    plt.savefig(output_path)

    logger.info(f"Plot saved to: {output_path}\n")

def compute_and_plot_correlations(df: pd.DataFrame):
    """
    Computes Pearson and Spearman correlations for numeric columns.
    Saves the matrices as CSV and heatmaps as PNG using academic styling.
    """
    logger.info("Computing correlations (Pearson & Spearman)...")
    utils.apply_plot_style()

    # Filter numeric columns only
    df_num = df.select_dtypes(include=[np.number])

    if df_num.empty:
        logger.warning("No numeric data found for correlation analysis.\n")
        return

    methods = ['pearson', 'spearman']

    for method in methods:
        logger.info(f"-> Calculating {method.capitalize()} correlation...")
        corr_matrix = df_num.corr(method=method)

        # Save CSV
        csv_path = config.OUTPUT_DIR / f"02_correlation_matrix_{method}.csv"
        corr_matrix.to_csv(csv_path)

        # Save Heatmap
        plt.figure(figsize=(12, 10))

        # Create a mask for the upper triangle (cleaner look for papers)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=False, # Removed annotations to avoid clutter on large matrices
            cmap='coolwarm', 
            vmax=1, 
            vmin=-1, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5}
        )

        plt.title(f'{method.capitalize()} Correlation Matrix', pad=20)
        plt.tight_layout()

        plot_path = config.OUTPUT_DIR / f"02_heatmap_{method}.png"
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Saved {method} matrix and heatmap.\n")

def compute_pps_matrix(df: pd.DataFrame):
    """
    Computes the Predictive Power Score (PPS) matrix.
    Uses 'magma' colormap for better contrast in publication.
    """
    logger.info("Computing PPS Matrix (Predictive Power Score)...")
    utils.apply_plot_style()

    try:
        # Calculate full PPS matrix
        matrix_df = pps.matrix(df)[['x', 'y', 'ppscore', 'case', 'is_valid_score']]

        # * Visualization 1: Full PPS Matrix (Features vs Features)
        logger.info("-> Generating Full PPS Matrix")
        pps_pivot = matrix_df.pivot(columns='x', index='y', values='ppscore')

        # Save data
        matrix_df.to_csv(config.OUTPUT_DIR / "02_pps_matrix_raw.csv", index=False)
        pps_pivot.to_csv(config.OUTPUT_DIR / "02_pps_matrix_pivot.csv")

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            pps_pivot, 
            annot=False, 
            cmap='magma',
            linewidths=0.5,
            square=True
        )
        plt.title('Predictive Power Score (PPS) Matrix - Full', pad=20)
        plt.tight_layout()

        plot_path = config.OUTPUT_DIR / "02_heatmap_pps_full.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved Full PPS matrix and heatmap.\n")

        # * Visualization 2: Target Predictors (Features vs Target)
        target = config.TARGET_COLUMN
        logger.info(f"-> Generating PPS for Target Column: {target}")

        target_predictors = matrix_df[
            (matrix_df['y'] == target) & 
            (matrix_df['x'] != target)
        ].copy()

        # Sort descending by predictive power
        target_predictors = target_predictors.sort_values('ppscore', ascending=False)

        # Take only top 20 predictors to avoid giant vertical plots
        if len(target_predictors) > 20:
             target_predictors = target_predictors.head(20)

        if not target_predictors.empty:
            # Pivot for heatmap (Index=Feature, Column=Target)
            pivot_target = target_predictors.set_index('x')[['ppscore']]

            plt.figure(figsize=(6, 10)) # Adjusted aspect ratio
            sns.heatmap(
                pivot_target, 
                annot=True, 
                fmt='.2f', 
                cmap='magma', 
                cbar_kws={'label': 'Predictive Power Score (PPS)'}, 
                linewidths=0.5, 
                annot_kws={'size': 10}
            )
            plt.title(f'Top Predictors for {target}', pad=20)
            plt.ylabel('Feature')
            plt.xlabel('Target')
            plt.tight_layout()

            plot_path_target = config.OUTPUT_DIR / "02_heatmap_pps_target_only.png"
            plt.savefig(plot_path_target)
            plt.close()

            logger.info(f"Saved Target-Specific PPS heatmap.\n")
        else:
            logger.warning(f"No predictors found for target {target} in PPS matrix.\n")

    except Exception as e:
        logger.error(f"Failed to compute PPS: {e}\n")
