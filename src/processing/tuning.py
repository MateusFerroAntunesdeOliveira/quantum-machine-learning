# Used for hyperparameter tuning of ML models using Optuna

import logging

import json
import pandas as pd

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import StratifiedKFold, cross_val_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.shared import config

logger = logging.getLogger(__name__)

def run_optimization(model_name: str, X: pd.DataFrame, y: pd.Series, n_trials: int = 50):
    """
    Runs hyperparameter optimization for a given model using Optuna.

    Args:
        model_name: String identifier for the model (e.g., 'LightGBM').
        X, y: Training data.
        n_trials: Number of optimization trials.

    Returns:
        None
    """
    logger.info(f'Starting optimization for {model_name} with {n_trials} trials...')

    # Suppress Optuna logging to avoid cluttering our logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # * Use Optuna to optimize hyperparameters - maximize cross-validated F1-score using TPE sampler (for reproducibility)
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))

    if model_name == 'LightGBM':
        study.optimize(lambda trial: objective_lightgbm(trial, X, y), n_trials=n_trials)
    elif model_name == 'XGBoost':
        study.optimize(lambda trial: objective_xgboost(trial, X, y), n_trials=n_trials)
    else:
        logger.error(f"Optimization for model '{model_name}' is not implemented.")
        return

    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"Optimization completed for {model_name}.")
    logger.info(f"Best CV F1-Score: {best_value:.4f}")
    logger.info(f"Best Hyperparameters: {best_params}")

    output_file = f"{config.BEST_PARAMS_FILE}_{model_name}.json"
    logger.info(f"Saving best hyperparameters to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=4)

    logger.info("Hyperparameter optimization process finished.\n")

def objective_lightgbm(trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Optuna objective function for LightGBM hyperparameter tuning.
    """
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        # Regularization (Important to avoid overfitting)
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        # Fixed
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': -1,
        # ...
    }
    model = LGBMClassifier(**param_grid)
    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cross_validator, scoring='f1')
    return scores.mean()

def objective_xgboost(trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Optuna objective function for XGBoost hyperparameter tuning.
    Defines the hyperparameter search space and evaluates using cross-validated F1-score.
    """
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        # Regularization (Important to avoid overfitting)
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0), # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0), # L2 regularization
        # Fixed
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
    }
    model = XGBClassifier(**param_grid)
    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cross_validator, scoring='f1')
    return scores.mean()
