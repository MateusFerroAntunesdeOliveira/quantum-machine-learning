# Used to generate SHAP explainability plots for the LightGBM model

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

import json
import logging
import shap

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lightgbm import LGBMClassifier

from src.shared import config

# Get logger instance for this module
logger = logging.getLogger(__name__)

def train_best_model(X: pd.DataFrame, y: pd.Series, best_params_path: str) -> LGBMClassifier:
    """
    Trains the final LightGBM model using the optimized hyperparameters found in Step 5.
    It fits on the entire dataset to maximize the information captured for SHAP analysis.
    """
    logger.info("Loading best hyperparameters from %s", best_params_path)
    with open(best_params_path, 'r') as file:
        best_params = json.load(file)

    logger.info(f"Training final LightGBM model with best hyperparameters: {best_params}")

    model = LGBMClassifier(**best_params, n_jobs=-1, verbose=-1, random_state=42)
    model.fit(X, y)
    logger.info("Final LightGBM model trained successfully.")
    return model

def perform_shap_analysis(model: LGBMClassifier, X: pd.DataFrame):
    """
    Performs SHAP analysis:
    1. Summary Plot (Beeswarm) - Global Importance
    2. Bar Plot - Mean Importance
    3. Interaction/Dependence Plots - Feature relationships
    """
    logger.info("Starting SHAP analysis...")

    # LightGBM is a tree-based model, so we use TreeExplainer (faster for trees)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle different SHAP versions return types (sometimes list, sometimes array)
    # For binary classification, LightGBM usually returns a list of [Class0, Class1] or just Class1
    if isinstance(shap_values, list):
        # We focus on the positive class (Autism = 1)
        shap_values_target = shap_values[1]
    else:
        shap_values_target = shap_values

    # --- PLOT 1: SUMMARY PLOT (Beeswarm) ---
    logger.info("Generating SHAP summary plot (Beeswarm)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_target, X, show=False, cmap='coolwarm')
    plt.title("SHAP Summary Plot (Global Interpretability)")
    plt.tight_layout()
    out_path = config.OUTPUT_DIR / "06_shap_summary_beeswarm.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"SHAP summary plot saved to {out_path}")

    # --- PLOT 2: BAR PLOT (Importance) ---
    logger.info("Generating SHAP mean importance bar plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_target, X, plot_type="bar", show=False, color='#4A90E2')
    plt.title("Mean Absolute SHAP Values (Feature Importance)")
    plt.tight_layout()
    out_path = config.OUTPUT_DIR / "06_shap_mean_importance_bar.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"SHAP mean importance bar plot saved to {out_path}")

    # --- PLOT 3: TOP 3 DEPENDENCE PLOTS (Interactions) ---
    mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:3]
    top_features = X.columns[top_indices]
    logger.info(f"Generating Dependence Plots for top features: {top_features.tolist()}...")
    for feat in top_features:
        plt.figure(figsize=(12, 8))
        # dependence_plot usually handles plotting itself, but we can capture it
        shap.dependence_plot(feat, shap_values_target, X, show=False, cmap='coolwarm', interaction_index='auto')
        plt.title(f"SHAP Dependence: {feat}")
        plt.tight_layout()
        safe_feat_name = feat.replace("^", "_pow_").replace(" ", "_x_")
        out_path_dep = config.OUTPUT_DIR / f"06_shap_dependence_{safe_feat_name}.png"
        plt.savefig(out_path_dep)
        plt.close()

