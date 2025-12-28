"""
Main training script for EPL In-Game Prediction
Train models with hyperparameter tuning and calibration
"""
import argparse
import logging
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from trainer import ModelTrainer
from models import get_base_classifiers
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
    
    args = parser.parse_args()
    
    logging.info("=" * 60)
    logging.info("EPL In-Game Prediction - Model Training")
    logging.info("=" * 60)
    
    # Load data
    loader = DataLoader()
    X, y, feature_names = loader.prepare_features_labels()
    
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
