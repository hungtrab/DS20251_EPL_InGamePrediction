"""
Feature engineering with Random Forest importance and SHAP values
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib
import os
import argparse
from config import RESULTS_DIR, RANDOM_SEED, NUM_CLASSES, TRAIN_DATA_PATH, TEST_DATA_PATH, FULL_DATA_PATH
from data_loader import DataLoader
from preprocessor import DataPreprocessor
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class FeatureImportanceAnalyzer:
    """Analyze feature importance using Random Forest and SHAP"""
    
    def __init__(self, feature_names):
        """
        Initialize analyzer
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        self.rf_model = None
        self.rf_importances = None
        self.shap_values = None
        self.shap_explainer = None
        
    def compute_rf_importance(self, X, y, n_estimators=500, max_depth=20, save_model=True):
        """
        Compute feature importance using Random Forest
        
        Args:
            X: Feature matrix
            y: Labels
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            save_model: Whether to save the trained model
        
        Returns:
            DataFrame with feature importances
        """
        logging.info("Computing Random Forest feature importance")
        logging.info(f"Training RF with {n_estimators} estimators, max_depth={max_depth}")
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        
        self.rf_model.fit(X, y)
        
        # Get feature importances
        importances = self.rf_model.feature_importances_
        
        # Create DataFrame
        self.rf_importances = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })
        
        self.rf_importances = self.rf_importances.sort_values('Importance', ascending=False)
        self.rf_importances['Cumulative_Importance'] = self.rf_importances['Importance'].cumsum()
        self.rf_importances['Rank'] = range(1, len(self.rf_importances) + 1)
        
        logging.info(f"Top 5 features by RF importance:")
        for idx, row in self.rf_importances.head(5).iterrows():
            logging.info(f"  {row['Rank']}. {row['Feature']}: {row['Importance']:.4f}")
        
        # Save results
        output_path = os.path.join(RESULTS_DIR, 'feature_importance', 'rf_importance.csv')
        self.rf_importances.to_csv(output_path, index=False)
        logging.info(f"RF importance saved to {output_path}")
        
        # Save model
        if save_model:
            model_path = os.path.join(RESULTS_DIR, 'models', 'rf_feature_importance.pkl')
            joblib.dump(self.rf_model, model_path)
            logging.info(f"RF model saved to {model_path}")
        
        return self.rf_importances
    
    def plot_rf_importance(self, top_n=20, figsize=(10, 8)):
        """
        Plot Random Forest feature importance
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size
        """
        if self.rf_importances is None:
            raise ValueError("RF importance not computed yet. Run compute_rf_importance first.")
        
        plt.figure(figsize=figsize)
        
        top_features = self.rf_importances.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Features by Random Forest Importance', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_path = os.path.join(RESULTS_DIR, 'figures', 'rf_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"RF importance plot saved to {output_path}")
        plt.close()
        
    def compute_shap_values(self, X, y=None, sample_size=1000, save_explainer=True):
        """
        Compute SHAP values for feature importance
        
        Args:
            X: Feature matrix
            y: Labels (optional, for training if model not already trained)
            sample_size: Number of samples to use for SHAP computation
            save_explainer: Whether to save the SHAP explainer
        
        Returns:
            SHAP values array
        """
        logging.info("Computing SHAP values")
        
        # Train model if not already trained
        if self.rf_model is None:
            if y is None:
                raise ValueError("Labels (y) required to train model")
            logging.info("Training RF model for SHAP analysis")
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
            self.rf_model.fit(X, y)
        
        # Subsample for SHAP (can be computationally expensive)
        if X.shape[0] > sample_size:
            logging.info(f"Subsampling {sample_size} instances for SHAP computation")
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create SHAP explainer
        logging.info("Creating SHAP TreeExplainer")
        self.shap_explainer = shap.TreeExplainer(self.rf_model)
        
        # Compute SHAP values
        logging.info("Computing SHAP values (this may take a while...)")
        self.shap_values = self.shap_explainer.shap_values(X_sample)
        
        # For multi-class, shap_values is a list
        if isinstance(self.shap_values, list):
            logging.info(f"Multi-class SHAP values computed for {len(self.shap_values)} classes")
        else:
            logging.info("Binary classification SHAP values computed")
        
        # Save SHAP values
        output_path = os.path.join(RESULTS_DIR, 'feature_importance', 'shap_values.npy')
        np.save(output_path, self.shap_values)
        logging.info(f"SHAP values saved to {output_path}")
        
        # Save explainer
        if save_explainer:
            explainer_path = os.path.join(RESULTS_DIR, 'models', 'shap_explainer.pkl')
            joblib.dump(self.shap_explainer, explainer_path)
            logging.info(f"SHAP explainer saved to {explainer_path}")
        
        return self.shap_values, X_sample
    
    def plot_shap_summary(self, X_sample, class_idx=0, plot_type='bar', max_display=20):
        """
        Plot SHAP summary
        
        Args:
            X_sample: Sample features used for SHAP computation
            class_idx: Class index for multi-class (0=Win, 1=Draw, 2=Loss)
            plot_type: Type of plot ('bar', 'dot', 'violin')
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Run compute_shap_values first.")
        
        class_names = ['Win', 'Draw', 'Loss']
        
        # Select SHAP values for the specified class
        if isinstance(self.shap_values, list):
            shap_values_class = self.shap_values[class_idx]
            title_suffix = f"({class_names[class_idx]})"
        else:
            shap_values_class = self.shap_values
            title_suffix = ""
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(
                shap_values_class,
                X_sample,
                feature_names=self.feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
            plt.title(f'SHAP Feature Importance {title_suffix}', fontsize=14)
        else:
            shap.summary_plot(
                shap_values_class,
                X_sample,
                feature_names=self.feature_names,
                plot_type=plot_type,
                max_display=max_display,
                show=False
            )
        
        plt.tight_layout()
        output_path = os.path.join(
            RESULTS_DIR, 'figures', 
            f'shap_summary_{plot_type}_class{class_idx}.png'
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"SHAP summary plot saved to {output_path}")
        plt.close()
    
    def get_shap_feature_importance(self, class_idx=0):
        """
        Get feature importance from SHAP values
        
        Args:
            class_idx: Class index for multi-class
        
        Returns:
            DataFrame with SHAP-based feature importance
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Run compute_shap_values first.")
        
        # Select SHAP values for the specified class
        if isinstance(self.shap_values, list):
            shap_values_class = self.shap_values[class_idx]
        else:
            # SHAP may return an Explanation object; extract raw array values
            shap_values_class = getattr(self.shap_values, 'values', self.shap_values)
        
        # Ensure we are working with a 2D numpy array of shape (n_samples, n_features)
        shap_array = np.asarray(shap_values_class)
        if shap_array.ndim == 3:
            # Handle multi-output arrays with shape (n_samples, n_features, n_outputs)
            # Select the requested class along the last axis
            if shap_array.shape[-1] >= 1:
                shap_array = shap_array[..., class_idx]
            # If squeezing still leaves an unexpected shape, fall through to the check below
        if shap_array.ndim != 2:
            raise ValueError(f"Unexpected SHAP values shape: {shap_array.shape}. Expected 2D (n_samples, n_features).")

        # Compute mean absolute SHAP values across samples -> shape (n_features,)
        mean_abs_shap = np.abs(shap_array).mean(axis=0)
        mean_abs_shap = np.asarray(mean_abs_shap).ravel()
        
        # Create DataFrame
        feature_list = list(self.feature_names)
        if len(feature_list) != mean_abs_shap.shape[0]:
            raise ValueError(
                f"Feature count ({len(feature_list)}) does not match SHAP importance length ({mean_abs_shap.shape[0]})."
            )
        shap_importance = pd.DataFrame({
            'Feature': feature_list,
            'SHAP_Importance': mean_abs_shap
        })
        
        shap_importance = shap_importance.sort_values('SHAP_Importance', ascending=False)
        shap_importance['Cumulative_Importance'] = shap_importance['SHAP_Importance'].cumsum()
        shap_importance['Rank'] = range(1, len(shap_importance) + 1)
        
        # Save
        output_path = os.path.join(
            RESULTS_DIR, 'feature_importance', 
            f'shap_importance_class{class_idx}.csv'
        )
        shap_importance.to_csv(output_path, index=False)
        logging.info(f"SHAP importance saved to {output_path}")
        
        return shap_importance


def main():
    """Main function for feature engineering"""
    parser = argparse.ArgumentParser(description='Feature Engineering for EPL Prediction')
    parser.add_argument('--mode', type=str, required=True, choices=['rf', 'shap'],
                       help='Mode: rf (Random Forest) or shap (SHAP values)')
    parser.add_argument('--data', type=str, default='train',
                       help='Data to use: train or test')
    parser.add_argument('--n_estimators', type=int, default=500,
                       help='Number of estimators for RF')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Sample size for SHAP computation')
    parser.add_argument('--top_n', type=int, default=20,
                       help='Number of top features to display')
    
    args = parser.parse_args()
    
    # Load data
    logging.info(f"Loading {args.data} data")
    if args.data == 'train':
        data_path = TRAIN_DATA_PATH
    elif args.data == 'test':
        data_path = TEST_DATA_PATH
    else:
        data_path = FULL_DATA_PATH
    loader = DataLoader(data_path=data_path)
    X, y, feature_names = loader.prepare_features_labels()
    
    # Preprocess
    preprocessor = DataPreprocessor(scaler_type='standard')
    X_scaled = preprocessor.fit_transform(X)
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(feature_names)
    
    if args.mode == 'rf':
        # Random Forest mode
        logging.info("=" * 60)
        logging.info("MODE: Random Forest Feature Importance")
        logging.info("=" * 60)
        
        # Compute importance
        analyzer.compute_rf_importance(X_scaled, y, n_estimators=args.n_estimators)
        
        # Plot
        analyzer.plot_rf_importance(top_n=args.top_n)
        
        logging.info("Random Forest feature importance analysis complete!")
        
    elif args.mode == 'shap':
        # SHAP mode
        logging.info("=" * 60)
        logging.info("MODE: SHAP Feature Importance")
        logging.info("=" * 60)
        
        # Compute SHAP values
        shap_values, X_sample = analyzer.compute_shap_values(
            X_scaled, y, 
            sample_size=args.sample_size
        )
        
        # Plot for each class
        for class_idx in range(NUM_CLASSES):
            logging.info(f"\nGenerating SHAP plots for class {class_idx}")
            analyzer.plot_shap_summary(X_sample, class_idx=class_idx, plot_type='bar')
            analyzer.plot_shap_summary(X_sample, class_idx=class_idx, plot_type='dot')
            
            # Get importance
            shap_imp = analyzer.get_shap_feature_importance(class_idx=class_idx)
            logging.info(f"\nTop 5 features for class {class_idx}:")
            for idx, row in shap_imp.head(5).iterrows():
                logging.info(f"  {row['Rank']}. {row['Feature']}: {row['SHAP_Importance']:.4f}")
        
        logging.info("\nSHAP feature importance analysis complete!")


if __name__ == '__main__':
    main()
