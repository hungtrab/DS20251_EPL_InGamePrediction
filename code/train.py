"""
Main training script for EPL In-Game Prediction
Train models with hyperparameter tuning and calibration
"""
import argparse
import logging
import os
from typing import List
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from trainer import ModelTrainer
from models import get_base_classifiers
from config import RESULTS_DIR
import glob
import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Train EPL Prediction Models')
    parser.add_argument('--evaluate-all', action='store_true',
                       help='Evaluate all base classifiers first')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['random_forest', 'xgboost'],
                       help='Models to train (space-separated)')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--calibrate', action='store_true', default=True,
                       help='Calibrate models')
    parser.add_argument('--scaler', type=str, default='standard',
                       choices=['standard', 'robust', 'minmax'],
                       help='Scaler type')
    parser.add_argument('--shap-top-k', type=int, default=20,
                       help='Use top-K features from SHAP/RF importances (set 0 to disable)')
    parser.add_argument('--importance-source', type=str, default='auto',
                       choices=['auto', 'shap', 'rf'],
                       help='Feature importance source: auto prefers SHAP, falls back to RF')
    
    args = parser.parse_args()
    
    logging.info("=" * 60)
    logging.info("EPL In-Game Prediction - Model Training")
    logging.info("=" * 60)
    
    # Load data
    loader = DataLoader()
    X, y, feature_names = loader.prepare_features_labels()
    original_feature_names = feature_names[:]

    # Optionally reduce to top-K features based on saved importances
    if args.shap_top_k and args.shap_top_k > 0:
        def load_top_k_features(k: int, source: str) -> List[str]:
            fi_dir = os.path.join(RESULTS_DIR, 'feature_importance')
            os.makedirs(fi_dir, exist_ok=True)

            def from_shap():
                paths = sorted(glob.glob(os.path.join(fi_dir, 'shap_importance_class*.csv')))
                if not paths:
                    return None
                dfs = []
                for p in paths:
                    try:
                        df = pd.read_csv(p)[['Feature', 'SHAP_Importance']]
                        dfs.append(df)
                    except Exception:
                        continue
                if not dfs:
                    return None
                merged = pd.concat(dfs, axis=0, ignore_index=True)
                agg = merged.groupby('Feature', as_index=False)['SHAP_Importance'].mean()
                agg = agg.sort_values('SHAP_Importance', ascending=False)
                return agg['Feature'].head(k).tolist()

            def from_rf():
                rf_path = os.path.join(fi_dir, 'rf_importance.csv')
                if not os.path.exists(rf_path):
                    return None
                try:
                    df = pd.read_csv(rf_path)[['Feature', 'Importance']]
                    df = df.sort_values('Importance', ascending=False)
                    return df['Feature'].head(k).tolist()
                except Exception:
                    return None

            top = None
            if source in ('auto', 'shap'):
                top = from_shap()
            if top is None and source in ('auto', 'rf'):
                top = from_rf()
            return top

        top_features = load_top_k_features(args.shap_top_k, args.importance_source)
        if top_features:
            # Map to indices and filter X and feature_names
            name_to_idx = {name: i for i, name in enumerate(feature_names)}
            selected_indices = [name_to_idx[f] for f in top_features if f in name_to_idx]
            if selected_indices:
                import numpy as np
                X = X[:, selected_indices]
                feature_names = [feature_names[i] for i in selected_indices]
                logging.info(f"Using top-{len(feature_names)} features from {args.importance_source} importance")
                logging.info(f"Selected features (first 10): {feature_names[:10]}")
            else:
                logging.warning("No overlap between SHAP/RF top features and loaded features. Using all features.")
        else:
            logging.warning("No SHAP/RF importance files found. Using all features.")
    
    # Preprocess
    preprocessor = DataPreprocessor(scaler_type=args.scaler)
    X_scaled = preprocessor.fit_transform(X)
    preprocessor.save()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Evaluate all base classifiers
    if args.evaluate_all:
        logging.info("\n" + "=" * 60)
        logging.info("Evaluating All Base Classifiers")
        logging.info("=" * 60)
        
        classifiers = get_base_classifiers()
        results = trainer.evaluate_base_classifiers(classifiers, X_scaled, y)
        
        print("\n" + results.to_string())
        
        # Save results
        results.to_csv('results/metrics/base_classifier_evaluation.csv', index=False)
        logging.info("\nResults saved to results/metrics/base_classifier_evaluation.csv")
    
    # Train selected models
    logging.info("\n" + "=" * 60)
    logging.info("Training Selected Models")
    logging.info("=" * 60)
    
    for model_name in args.models:
        logging.info(f"\n{'=' * 40}")
        logging.info(f"Model: {model_name}")
        logging.info(f"{'=' * 40}")
        
        try:
            # Train and calibrate
            model, calibrated_model = trainer.train_and_calibrate(
                model_name, 
                X_scaled, 
                y, 
                tune_hyperparams=args.tune
            )
            
            # Save models
            trainer.save_model(model, model_name)
            
            if args.calibrate:
                trainer.save_model(calibrated_model, model_name, suffix='_calibrated')
            
        except Exception as e:
            logging.error(f"Error training {model_name}: {e}")
            continue
    
    # Save training results
    trainer.save_results()
    
    logging.info("\n" + "=" * 60)
    logging.info("Training Complete!")
    logging.info("=" * 60)
    logging.info(f"Models saved to: results/models/")
    logging.info(f"Metrics saved to: results/metrics/")


if __name__ == '__main__':
    main()
