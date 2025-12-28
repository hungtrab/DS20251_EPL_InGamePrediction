"""
Training utilities for EPL In-Game Prediction
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from skopt import BayesSearchCV
import joblib
import os
import json
from datetime import datetime
from config import (
    RESULTS_DIR, RANDOM_SEED, CV_SPLITS, CALIB_SIZE,
    BAYESIAN_SEARCH_PARAMS, N_ITER_PARAMS
)
from metrics import rps_scorer, compute_all_metrics
from models import get_classifier_by_name, calibrate_classifier
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class ModelTrainer:
    """Model training and hyperparameter optimization"""
    
    def __init__(self):
        self.results = {}
        self.trained_models = {}
        
    def evaluate_base_classifiers(self, classifiers, X, y, n_splits=10):
        """
        Evaluate multiple classifiers using stratified shuffle split
        
        Args:
            classifiers: List of classifier instances
            X: Features
            y: Labels
            n_splits: Number of splits for cross-validation
        
        Returns:
            DataFrame with results
        """
        logging.info(f"Evaluating {len(classifiers)} base classifiers")
        
        acc_dict = {}
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=RANDOM_SEED)
        
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            for clf in classifiers:
                name = clf.__class__.__name__
                
                clf.fit(X_train, y_train)
                predictions = clf.predict_proba(X_test)
                metrics = compute_all_metrics(y_test, predictions)
                
                if name not in acc_dict:
                    acc_dict[name] = {k: [] for k in metrics.keys()}
                
                for k, v in metrics.items():
                    acc_dict[name][k].append(v)
        
        # Average results
        results = []
        for name, metrics in acc_dict.items():
            result = {'Classifier': name}
            for metric_name, values in metrics.items():
                result[metric_name] = np.mean(values)
                result[f'{metric_name}_std'] = np.std(values)
            results.append(result)
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='rps_score', ascending=False)
        
        logging.info("Base classifier evaluation complete")
        return df_results
    
    def hyperparameter_tuning(self, model_name, X, y, param_grid=None, n_iter=None):
        """
        Perform Bayesian hyperparameter optimization
        
        Args:
            model_name: Name of the model to tune
            X: Features
            y: Labels
            param_grid: Parameter grid (optional, uses default if None)
            n_iter: Number of iterations (optional, uses default if None)
        
        Returns:
            Best estimator and best score
        """
        if param_grid is None:
            if model_name not in BAYESIAN_SEARCH_PARAMS:
                raise ValueError(f"No default param grid for {model_name}")
            param_grid = BAYESIAN_SEARCH_PARAMS[model_name]
        
        if n_iter is None:
            n_iter = N_ITER_PARAMS.get(model_name, 50)
        
        logging.info(f"Starting hyperparameter tuning for {model_name}")
        logging.info(f"Iterations: {n_iter}, CV splits: {CV_SPLITS}")
        
        # Get base classifier
        base_clf = get_classifier_by_name(model_name)
        
        # Bayesian optimization
        bayes_search = BayesSearchCV(
            base_clf,
            param_grid,
            n_iter=n_iter,
            scoring=rps_scorer,
            cv=CV_SPLITS,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=1
        )
        
        bayes_search.fit(X, y)
        
        logging.info(f"Best score: {bayes_search.best_score_:.4f}")
        logging.info(f"Best params: {bayes_search.best_params_}")
        
        self.results[model_name] = {
            'best_score': bayes_search.best_score_,
            'best_params': bayes_search.best_params_,
            'cv_results': bayes_search.cv_results_
        }
        
        return bayes_search.best_estimator_, bayes_search.best_score_
    
    def train_and_calibrate(self, model_name, X, y, tune_hyperparams=True):
        """
        Train a model with optional hyperparameter tuning and calibration
        
        Args:
            model_name: Name of the model
            X: Features
            y: Labels
            tune_hyperparams: Whether to tune hyperparameters
        
        Returns:
            Tuple of (trained_model, calibrated_model)
        """
        logging.info(f"Training {model_name}")
        
        # Split data for training and calibration
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, y, test_size=CALIB_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        
        # Train model
        if tune_hyperparams:
            model, score = self.hyperparameter_tuning(model_name, X_train, y_train)
        else:
            model = get_classifier_by_name(model_name)
            model.fit(X_train, y_train)
            predictions = model.predict_proba(X_calib)
            metrics = compute_all_metrics(y_calib, predictions)
            score = metrics['rps_score']
        
        # Calibrate model
        calibrated_model = calibrate_classifier(model, X_calib, y_calib)
        
        # Store models
        self.trained_models[model_name] = model
        self.trained_models[f'{model_name}_calibrated'] = calibrated_model
        
        return model, calibrated_model
    
    def save_model(self, model, name, suffix=''):
        """Save trained model"""
        filename = f"{name}{suffix}.pkl"
        filepath = os.path.join(RESULTS_DIR, 'models', filename)
        joblib.dump(model, filepath)
        logging.info(f"Model saved to {filepath}")
        
    def save_results(self, filename='training_results.json'):
        """Save training results"""
        filepath = os.path.join(RESULTS_DIR, 'metrics', filename)
        
        # Convert numpy types to native Python types for JSON serialization
        results_serializable = {}
        for model_name, result in self.results.items():
            results_serializable[model_name] = {
                'best_score': float(result['best_score']),
                'best_params': {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) 
                               for k, v in result['best_params'].items()}
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logging.info(f"Results saved to {filepath}")
