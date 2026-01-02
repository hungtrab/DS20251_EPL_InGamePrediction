"""
Main training script for EPL In-Game Prediction
Train models with hyperparameter tuning and calibration
"""
import argparse
import logging
import os
from typing import List
import numpy as np
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from trainer import ModelTrainer
from models import get_base_classifiers
from config import RESULTS_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH, FULL_DATA_PATH
import glob
from metrics import compute_all_metrics
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
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='epl-in-game-prediction', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (username or team)')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
    parser.add_argument('--wandb-tags', type=str, nargs='*', default=None, help='W&B tags')
    parser.add_argument('--wandb-mode', type=str, default='online', choices=['online','offline','disabled'], help='W&B mode')
    parser.add_argument('--wandb-notes', type=str, default=None, help='W&B run notes')
    parser.add_argument('--data', type=str, default='train', choices=['train','test','full'],
                        help='Which dataset to use')
    
    args = parser.parse_args()
    
    logging.info("=" * 60)
    logging.info("EPL In-Game Prediction - Model Training")
    logging.info("=" * 60)
    
    # Load data
    # Resolve data path
    if args.data == 'train':
        data_path = TRAIN_DATA_PATH
    elif args.data == 'test':
        data_path = TEST_DATA_PATH
    else:
        data_path = FULL_DATA_PATH
    logging.info(f"Loading dataset: {args.data} ({data_path})")

    loader = DataLoader(data_path=data_path)
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
                logging.info(
                    f"Using top-{len(feature_names)} features from {args.importance_source} importance"
                )
                logging.info(f"Selected features (top-{args.shap_top_k}): {feature_names}")
            else:
                logging.warning("No overlap between SHAP/RF top features and loaded features. Using all features.")
        else:
            logging.warning("No SHAP/RF importance files found. Using all features.")
    
    # Preprocess
    preprocessor = DataPreprocessor(scaler_type=args.scaler)
    X_scaled = preprocessor.fit_transform(X)
    preprocessor.save()

    # Optional: initialize Weights & Biases
    wandb_run = None
    if args.wandb and args.wandb_mode != 'disabled':
        try:
            import wandb
            if args.wandb_mode == 'offline':
                os.environ['WANDB_MODE'] = 'offline'
            wandb_login_ok = True
            # Initialize run
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                tags=args.wandb_tags,
                notes=args.wandb_notes,
                config={
                    'models': args.models,
                    'tune': args.tune,
                    'calibrate': args.calibrate,
                    'scaler': args.scaler,
                    'shap_top_k': args.shap_top_k,
                    'importance_source': args.importance_source,
                    'n_samples': int(X.shape[0]),
                    'n_features': int(X.shape[1]),
                    'label_distribution': np.bincount(y).tolist(),
                }
            )
            # Log selected features if reduced
            if len(feature_names) != len(original_feature_names):
                try:
                    import wandb as _wandb
                    _wandb.config.update({'selected_features': feature_names}, allow_val_change=True)
                except Exception:
                    pass
        except Exception as e:
            logging.warning(f"W&B initialization failed: {e}. Continuing without W&B.")
            wandb_run = None
    
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
        # Log to W&B
        if wandb_run is not None:
            try:
                import wandb
                table = wandb.Table(dataframe=results)
                wandb.log({'base_classifier_evaluation': table})
            except Exception as e:
                logging.warning(f"W&B log (base eval) failed: {e}")
    
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

            # Log metrics & artifacts to W&B
            if wandb_run is not None:
                try:
                    import wandb
                    # Log best params/score if tuning
                    if model_name in trainer.results:
                        res = trainer.results[model_name]
                        wandb.log({
                            f'{model_name}/best_rps_cv': float(res.get('best_score', np.nan))
                        })
                        best_params = res.get('best_params', {})
                        for k, v in best_params.items():
                            wandb.log({f'{model_name}/best_params/{k}': v})
                    # Evaluate on full train-calib set (proxy)
                    try:
                        preds = calibrated_model.predict_proba(X_scaled)
                        m = compute_all_metrics(y, preds)
                        wandb.log({f'{model_name}/rps_train': m.get('rps_score', np.nan)})
                        for mk, mv in m.items():
                            if mk == 'rps_score':
                                continue
                            wandb.log({f'{model_name}/{mk}_train': mv})
                    except Exception as eval_e:
                        logging.warning(f"Train-set metric logging failed for {model_name}: {eval_e}")
                    # Log model artifacts
                    try:
                        model_path = os.path.join(RESULTS_DIR, 'models', f"{model_name}.pkl")
                        if os.path.exists(model_path):
                            art = wandb.Artifact(f'{model_name}-model', type='model')
                            art.add_file(model_path)
                            wandb.log_artifact(art)
                        calib_path = os.path.join(RESULTS_DIR, 'models', f"{model_name}_calibrated.pkl")
                        if os.path.exists(calib_path):
                            artc = wandb.Artifact(f'{model_name}-model-calibrated', type='model')
                            artc.add_file(calib_path)
                            wandb.log_artifact(artc)
                    except Exception as art_e:
                        logging.warning(f"W&B artifact logging failed for {model_name}: {art_e}")
                except Exception as we:
                    logging.warning(f"W&B logging failed for {model_name}: {we}")
            
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

    # Finish W&B run
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
