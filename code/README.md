# Modular EPL In-Game Prediction

Refactored modular structure for English Premier League in-game prediction system.

## 📁 Project Structure

```
code/
├── config.py                 # Configuration and constants
├── data_loader.py            # Data loading utilities
├── preprocessor.py           # Data preprocessing and feature engineering
├── metrics.py                # Custom evaluation metrics (RPS, Brier, etc.)
├── models.py                 # Model definitions
├── trainer.py                # Training and hyperparameter optimization
├── feature_engineering.py    # Feature importance analysis (RF & SHAP)
├── data_integration.py       # Convert event data to match statistics
├── data_cleaning.py          # Clean and split data
└── train.py                  # Main training script

results/
├── models/                   # Trained models
├── figures/                  # Visualizations
├── metrics/                  # Training metrics
└── feature_importance/       # Feature analysis results
```

## 🚀 Quick Start

### 1. Data Pipeline

```bash
cd code

# Integrate pregame + event data
python data_integration.py

# Clean and split into train/test
python data_cleaning.py --test-ratio 0.2
```

### 2. Feature Engineering

```bash
# Random Forest feature importance
python feature_engineering.py --mode rf --n_estimators 500

# SHAP values analysis
python feature_engineering.py --mode shap --sample_size 1000
```

### 3. Model Training

```bash
# Evaluate all base classifiers
python train.py --evaluate-all

# Train specific models with tuning
python train.py --models random_forest xgboost --tune --calibrate

# Quick training without tuning
python train.py --models random_forest
```

## 📊 Usage Examples

### Load and Preprocess Data

```python
from data_loader import DataLoader
from preprocessor import DataPreprocessor

# Load data
loader = DataLoader()
X, y, feature_names = loader.prepare_features_labels()

# Preprocess
preprocessor = DataPreprocessor(scaler_type='standard')
X_scaled = preprocessor.fit_transform(X)
preprocessor.save()
```

### Train a Model

```python
from trainer import ModelTrainer

trainer = ModelTrainer()

# Train with hyperparameter tuning
model, calibrated_model = trainer.train_and_calibrate(
    'xgboost', X_scaled, y, tune_hyperparams=True
)

# Save models
trainer.save_model(model, 'xgboost')
trainer.save_model(calibrated_model, 'xgboost', suffix='_calibrated')
```

### Feature Importance Analysis

```python
from feature_engineering import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(feature_names)

# Random Forest importance
rf_importance = analyzer.compute_rf_importance(X_scaled, y)
analyzer.plot_rf_importance(top_n=20)

# SHAP values
shap_values, X_sample = analyzer.compute_shap_values(X_scaled, y)
for class_idx in range(3):
    analyzer.plot_shap_summary(X_sample, class_idx=class_idx)
```

## 🔧 Configuration

Edit `config.py` to customize:
- Random seed
- Train/test split ratio
- Hyperparameter search grids
- Number of CV folds
- Directory paths

## 📈 Available Models

- K-Nearest Neighbors (knn)
- Decision Tree (decision_tree)
- Random Forest (random_forest)
- XGBoost (xgboost)
- AdaBoost (adaboost)
- Gradient Boosting (gradient_boosting)
- Logistic Regression (logistic_regression)
- Naive Bayes (naive_bayes)
- Linear/Quadratic Discriminant Analysis (lda, qda)

## 📏 Metrics

- **RPS (Ranked Probability Score)**: Primary metric for probabilistic predictions
- **Brier Score**: Multi-class calibration metric
- **Log Loss**: Cross-entropy loss
- **Accuracy**: Classification accuracy
- **ECE/MCE**: Expected/Maximum Calibration Error

## 🎯 Advanced Features

### Custom Feature Engineering

```python
from preprocessor import AdvancedFeatureProcessor

processor = AdvancedFeatureProcessor(feature_names)

# Create features
df = processor.create_interaction_features(data)
df = processor.create_temporal_features(df)
df = processor.create_momentum_features(df)
df = processor.create_rolling_features(df, window_sizes=[3, 5, 10])
```

### Command-Line Options

**Data Integration:**
```bash
python data_integration.py --overwrite  # Reprocess all matches
```

**Data Cleaning:**
```bash
python data_cleaning.py --test-ratio 0.25 --random-seed 123
```

**Feature Engineering:**
```bash
python feature_engineering.py --mode rf --n_estimators 1000 --top_n 15
python feature_engineering.py --mode shap --sample_size 2000
```

**Training:**
```bash
python train.py --evaluate-all --models knn decision_tree random_forest xgboost --tune
```

## 📦 Dependencies

```bash
pip install pandas numpy scikit-learn xgboost scikit-optimize shap tqdm
```

## 🔬 Results Structure

```
results/
├── models/
│   ├── preprocessor.pkl
│   ├── random_forest.pkl
│   ├── random_forest_calibrated.pkl
│   ├── xgboost.pkl
│   ├── xgboost_calibrated.pkl
│   ├── rf_feature_importance.pkl
│   └── shap_explainer.pkl
├── figures/
│   ├── rf_importance.png
│   ├── shap_summary_bar_class0.png
│   ├── shap_summary_bar_class1.png
│   └── shap_summary_bar_class2.png
├── metrics/
│   ├── training_results.json
│   └── base_classifier_evaluation.csv
└── feature_importance/
    ├── rf_importance.csv
    ├── shap_values.npy
    ├── shap_importance_class0.csv
    ├── shap_importance_class1.csv
    └── shap_importance_class2.csv
```

## 🎓 Key Improvements

✅ **Modular Design**: Separated concerns into focused modules  
✅ **Configuration Management**: Centralized settings  
✅ **Advanced Preprocessing**: Rolling features, interactions, temporal patterns  
✅ **Feature Analysis**: Dual-mode RF and SHAP importance  
✅ **Organized Results**: Structured output in `./results/`  
✅ **Reproducibility**: Consistent random seeds and saved configs  
✅ **Extensibility**: Easy to add new models, metrics, or features  
✅ **CLI Support**: Full command-line interface  
✅ **Logging**: Comprehensive progress tracking  

## 📝 Notes

- All scripts auto-detect project directory
- Results are saved to `./results/` by default
- Models are automatically calibrated using isotonic regression
- SHAP analysis can be computationally expensive (use --sample_size to control)

## 🔗 Integration with Existing Notebooks

The original notebooks (`demo.ipynb`, `evaluation.ipynb`, etc.) can still be used alongside the new modular code. Simply import the new modules:

```python
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from trainer import ModelTrainer
# ... etc
```
