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

from src.shared import config, utils

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
    utils.apply_plot_style()

    # LightGBM is a tree-based model, so we use TreeExplainer (faster for trees)
    explainer = shap.TreeExplainer(model)
    shap_values_obj = explainer(X)

    # Check shape mechanism for binary classification
    # shap_values_obj usually has shape (n_samples, n_features, n_classes) or (n_samples, n_features)
    # We focus on Class 1 (Autism)
    if len(shap_values_obj.shape) == 3:
        shap_values_target = shap_values_obj[:, :, 1]
    else:
        shap_values_target = shap_values_obj

    # --- PLOT 1: SUMMARY PLOT (Beeswarm) ---
    logger.info("Generating SHAP summary plot (Beeswarm)...")
    plt.figure()
    shap.plots.beeswarm(shap_values_target, show=False, max_display=15)
    plt.title("SHAP Summary Plot (Global Interpretability)", pad=20)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "06_shap_summary_beeswarm.png")
    plt.close()

    # --- PLOT 2: BAR PLOT (Importance) ---
    logger.info("Generating SHAP Bar Plot...")
    plt.figure()
    shap.plots.bar(shap_values_target, show=False, max_display=15)
    plt.title("Mean Feature Importance", pad=20)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "06_shap_mean_importance_bar.png")
    plt.close()

    # --- PLOT 3: WATERFALL PLOT (Local Explanation - Single Case) ---
    # Let's pick a high-confidence ASD case (e.g., the first one predicted as 1)
    # Or simply the first sample in X for demonstration
    sample_idx = 0 
    logger.info(f"Generating Waterfall Plot for Sample index {sample_idx}...")
    plt.figure()
    # Note: shap.plots.waterfall takes a single explanation object slice
    shap.plots.waterfall(shap_values_target[sample_idx], show=False, max_display=10)
    plt.title(f"Local Explanation (Waterfall) - Patient #{sample_idx}", pad=20)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "06_shap_waterfall_single_case.png")
    plt.close()

    # --- PLOT 4: DECISION PLOT (Path of decision) ---
    logger.info("Generating Decision Plot...")
    plt.figure()
    # We plot the first 20 observations to avoid clutter
    shap.plots.decision(
        shap_values_target[0].base_values, 
        shap_values_target.values[:20], 
        X.iloc[:20], 
        show=False
    )
    plt.title("Decision Path (First 20 Patients)", pad=20)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "06_shap_decision_path.png")
    plt.close()

    # --- PLOT 5: TOP 3 DEPENDENCE PLOTS (Interactions) ---
    mean_abs_shap = np.abs(shap_values_target.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:3]
    top_features = X.columns[top_indices]

    logger.info(f"Generating Dependence Plots for: {top_features.tolist()}...")
    for feat in top_features:
        plt.figure()
        shap.plots.scatter(shap_values_target[:, feat], color=shap_values_target, show=False)
        plt.title(f"Dependence Plot: {feat}", pad=20)
        plt.tight_layout()
        
        safe_feat_name = feat.replace("^", "_pow_").replace(" ", "_x_")
        plt.savefig(config.OUTPUT_DIR / f"06_shap_dependence_{safe_feat_name}.png")
        plt.close()
