"""
Configuration file for EPL In-Game Prediction project
"""
import os
import numpy as np

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Directory paths
PRJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PRJ_DIR, 'data')
MODEL_DIR = os.path.join(PRJ_DIR, 'model')
RESULTS_DIR = os.path.join(PRJ_DIR, 'results')

# Data paths
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train', 'data.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test', 'data.csv')
FULL_DATA_PATH = os.path.join(DATA_DIR, 'full.csv')

# Create results directory structure
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'feature_importance'), exist_ok=True)

# Model parameters
RESULT_MAP = {'W': 0, 'D': 1, 'L': 2}
NUM_CLASSES = 3
TEST_SIZE = 0.2
CALIB_SIZE = 0.2
CV_SPLITS = 5

# Hyperparameter search settings (from modelling.ipynb)
BAYESIAN_SEARCH_PARAMS = {
    'knn': {
        'n_neighbors': (3, 20),
        'weights': ['uniform'],  # notebook only uses uniform
        'metric': ['minkowski', 'euclidean', 'manhattan']
    },
    'decision_tree': {
        'max_depth': (3, 40),
        'min_samples_split': (2, 40),
        'min_samples_leaf': (1, 20),
        'max_features': [None, "sqrt", "log2"],
        'max_leaf_nodes': (10, 200),
        'min_impurity_decrease': (1e-5, 0.5, "uniform"),
        'class_weight': [None, "balanced"]
    },
    'random_forest': {
        'n_estimators': (100, 1000),  # notebook: step of 50, but skopt handles this
        'max_depth': (10, 100),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 5),
        'max_features': (0.1, 1.0),  # notebook uses float range
        'bootstrap': [True, False]
    },
    'xgboost': {
        'max_depth': (3, 10),
        'n_estimators': (100, 1000),  # notebook: step of 50
        'learning_rate': (0.01, 0.3),
        'colsample_bytree': (0.5, 1.0),
        'subsample': (0.6, 1.0)
    },
    'gradient_boosting': {
        'n_estimators': (50, 500),
        'max_depth': (3, 15),
        'learning_rate': (0.01, 0.3),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'subsample': (0.6, 1.0)
    },
    'adaboost': {
        'n_estimators': (50, 500),
        'learning_rate': (0.01, 2.0)
    }
}

# Number of iterations for Bayesian optimization
N_ITER_PARAMS = {
    'knn': 50,
    'decision_tree': 100,
    'random_forest': 50,
    'xgboost': 100,
    'gradient_boosting': 50,
    'adaboost': 50
}
