# * Used for Pearson, Spearman and PPS (Predictive Power Score) calculations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ppscore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import logging
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ppscore as pps
import seaborn as sns

from src.shared import config, translate, utils

# Get logger instance for this module
logger = logging.getLogger(__name__)

# --- PRIVATE HELPER FUNCTIONS (DRY Principle) ---

def _save_plot(filename: str):
    """
    Internal helper to save the current matplotlib figure using config paths.
    """
    output_path = config.OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=config.VIZ_PARAMS['figure.dpi'])
    plt.close()
    logger.info(f"Plot saved to: {output_path}\n")

def _plot_generic_heatmap(data: pd.DataFrame, title: str, filename: str, cmap: str, mask=None, annot=False, cbar_label=None):
    """
    Generic function to plot standardized heatmaps (Corr or PPS).
    """
    plt.figure(figsize=(12, 10))
    is_diverging = cmap == 'RdBu'

    sns.heatmap(
        data,
        mask=mask,
        annot=annot,
        cmap=cmap,
        vmax=1,
        vmin=-1 if is_diverging else 0,         # If PPS (Blues), start at 0
        center=0 if is_diverging else None,     # PPS does not need centering
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5, "label": cbar_label}
    )
    plt.title(title, pad=20)
    _save_plot(filename)

def _assign_group(feature_name: str) -> str:
    """
    Maps a feature name to a readable group based on regex patterns in config.
    """
    # Verify if GROUP_PATTERNS exists in config, else fallback or use empty dict
    patterns = getattr(config, 'GROUP_PATTERNS', {})

    for pattern, group_name in patterns.items():
        if pattern in feature_name:
            return group_name
        # Handle Regex cases
        if '.*' in pattern and re.search(pattern, feature_name):
            return group_name

    return feature_name

# --- PUBLIC FUNCTIONS ---

def plot_missing_distribution(report_df: pd.DataFrame, fileName: str):
    """
    Generates a horizontal bar plot of missing values (Grouped).
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
    df_plot['feature_group'] = df_plot.index.map(_assign_group)
    df_grouped = df_plot.groupby('feature_group')['pct'].mean().sort_values(ascending=False)
    top_missing = df_grouped.head(20)

    if top_missing.empty:
        logger.warning("No columns with missing values found to plot.\n")
        return

    plt.figure(figsize=(10, 8))
    barplot = sns.barplot(
        x=top_missing.values,
        y=top_missing.index,
        color='#4A90E2',
        edgecolor='none'
    )
    plt.title(translate.PLOT_LABELS['missing_title'], pad=20)
    plt.xlabel(translate.PLOT_LABELS['missing_xlabel'])
    plt.ylabel('')
    plt.xlim(0, 110) # Space for labels
    plt.grid(axis='x', alpha=0.3)

    for i, v in enumerate(top_missing.values):
        barplot.text(v + 1, i + 0.25, f"{v:.1f}%", color='black', fontsize=10)
    sns.despine()
    _save_plot(fileName)

def plot_class_balance(df: pd.DataFrame, target_col: str):
    """
    Plots the distribution of the target variable (Autism vs Control).
    """
    logger.info("Generating Class Balance Plot...")
    utils.apply_plot_style()

    if target_col not in df.columns:
        logger.warning(f"Target column {target_col} not found.")
        return

    counts = df[target_col].value_counts(normalize=True) * 100
    counts_raw = df[target_col].value_counts()

    # Map labels if possible (assuming 1=Autism, 2=Control based on Config)
    # We create a display mapping for the plot
    plot_data = pd.DataFrame({
        'Label': [translate.PLOT_LABELS['class_labels'].get(x, str(x)) for x in counts.index],
        'Percentage': counts.values,
        'Count': counts_raw.values,
        'Original': counts.index
    })

    plt.figure(figsize=(8, 6))
    bars = sns.barplot(
        data=plot_data,
        x='Label',
        y='Percentage',
        hue='Original',
        palette=['#E74C3C', '#4A90E2'], # Red (ASD), Blue (Control)
        edgecolor='black',
        legend=False
    )

    # Annotations
    for i, row in plot_data.iterrows():
        text = f"{row['Percentage']:.1f}%\n(N={row['Count']})"
        bars.text(i, row['Percentage'] + 1, text, ha='center', color='black', fontsize=12)

    plt.title(translate.PLOT_LABELS['class_balance_title'], pad=20)
    plt.ylabel(translate.PLOT_LABELS['class_balance_ylabel'])
    plt.xlabel('')
    plt.ylim(0, 100)
    sns.despine()
    _save_plot(config.CLASS_BALANCE_PLOT.name)

def plot_stratified_missingness(df: pd.DataFrame, target_col: str, features_to_check: list):
    """
    Plots missingness percentage separated by Target Class.
    """
    logger.info("Generating Stratified Missingness Plot...")
    utils.apply_plot_style()
    
    if target_col not in df.columns:
        logger.warning(f"Target column {target_col} not found.")
        return

    plot_data = []
    for feat in features_to_check:
        if feat not in df.columns: continue

        missing_by_class = df.groupby(target_col)[feat].apply(lambda x: x.isnull().mean() * 100)

        for class_id, pct in missing_by_class.items():
            plot_data.append({
                'Feature': feat,
                'Class': translate.PLOT_LABELS['class_labels'].get(class_id, str(class_id)),
                'Missing %': pct,
                'Original_Class_Id': class_id 
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
        hue='Class',
        palette=['#4A90E2', '#E74C3C'], # Blue (Control), Red (ASD) - Check mapping order!
        edgecolor='black'
    )

    plt.title(translate.PLOT_LABELS['stratified_title'], pad=20)
    plt.ylabel(translate.PLOT_LABELS['stratified_ylabel'])
    plt.xlabel('')
    plt.ylim(0, 105)
    plt.legend(title='Grupo')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    _save_plot(config.MISSINGNESS_STRATIFIED_PLOT.name)

def compute_correlations(df: pd.DataFrame):
    """
    Computes Pearson and Spearman correlations and plots standardized heatmaps.
    """
    logger.info("Computing correlations...")
    utils.apply_plot_style()

    df_num = df.select_dtypes(include=[np.number])
    if df_num.empty:
        logger.warning("No numeric data found.")
        return

    methods = {
        'pearson': (config.CORR_MATRIX_PEARSON_CSV, config.HEATMAP_PLOT_PEARSON.name, translate.PLOT_LABELS['corr_pearson_title']),
        'spearman': (config.CORR_MATRIX_SPEARMAN_CSV, config.HEATMAP_PLOT_SPEARMAN.name, translate.PLOT_LABELS['corr_spearman_title'])
    }

    for method, (csv_path, plot_filename, title) in methods.items():
        logger.info(f"-> Calculating {method.capitalize()}...")
        
        corr_matrix = df_num.corr(method=method)
        corr_matrix.to_csv(csv_path)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        _plot_generic_heatmap(
            data=corr_matrix,
            title=title,
            filename=plot_filename,
            cmap='RdBu',
            mask=mask,
            annot=False # Too dense for full matrix
        )

def compute_pps(df: pd.DataFrame):
    """
    Computes PPS and generates full matrix and target-specific plots.
    """
    logger.info("Computing PPS Matrix...")
    utils.apply_plot_style()

    try:
        matrix_df = pps.matrix(df)[['x', 'y', 'ppscore', 'case', 'is_valid_score']]
        pps_pivot = matrix_df.pivot(columns='x', index='y', values='ppscore')

        # Save Data
        matrix_df.to_csv(config.CORR_MATRIX_PPS_CSV, index=False)
        pps_pivot.to_csv(config.CORR_MATRIX_PPS_PIVOT_CSV)

        # * Visualization 1: Full PPS Matrix (Features vs Features)
        _plot_generic_heatmap(
            data=pps_pivot,
            title=translate.PLOT_LABELS['pps_full_title'],
            filename=config.HEATMAP_PPS_FULL_PLOT.name,
            cmap='Blues', 
            cbar_label=translate.PLOT_LABELS['pps_cbar']
        )

        # * Visualization 2: Target Predictors (Features vs Target)
        target = config.TARGET_COLUMN
        logger.info(f"-> Generating PPS for Target: {target}")

        target_predictors = matrix_df[
            (matrix_df['y'] == target) & 
            (matrix_df['x'] != target)
        ].sort_values('ppscore', ascending=False).head(20)

        if not target_predictors.empty:
            pivot_target = target_predictors.set_index('x')[['ppscore']]

            # Custom plot for the strip (vertical)
            plt.figure(figsize=(6, 10))
            sns.heatmap(
                pivot_target,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                cbar_kws={'label': translate.PLOT_LABELS['pps_cbar']},
                linewidths=0.5,
                annot_kws={'size': 10},
                vmin=0,
                vmax=1
            )

            # Formata título com o nome da feature alvo
            plt.title(translate.PLOT_LABELS['pps_target_title'].format(target), pad=20)
            plt.ylabel(translate.PLOT_LABELS['pps_ylabel'])
            plt.xlabel(translate.PLOT_LABELS['pps_xlabel'])
            _save_plot(config.HEATMAP_PPS_TARGET_PLOT.name)
        else:
            logger.warning(f"No predictors found for {target}.\n")

    except Exception as e:
        logger.error(f"Failed to compute PPS: {e}\n")
