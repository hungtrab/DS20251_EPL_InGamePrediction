# EPL In-Game Prediction - Complete Usage Guide

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Full Pipeline Execution
```bash
# Step 1: Data Integration (if not done)
python code/data_integration.py --project-dir .

# Step 2: Data Cleaning & Splitting
python code/data_cleaning.py --test-ratio 0.2

# Step 3: Feature Engineering Analysis
python code/feature_engineering.py --mode rf --top-n 20

# Step 4: Train Models
python code/train.py --evaluate-all --tune --calibrate
```

---

## 📊 Module-by-Module Guide

### 1️⃣ Data Integration (`data_integration.py`)
**Purpose**: Convert raw event data to match-level statistics

**Usage**:
```bash
# Default (process all matches)
python code/data_integration.py

# Custom project directory
python code/data_integration.py --project-dir /path/to/project

# Overwrite existing files
python code/data_integration.py --overwrite
```

**What it does**:
- Reads event data from `data/event_data/*.csv`
- Processes events: goals, shots, cards, substitutions, etc.
- Creates match files in `data/match/`
- Shows progress bar for each match

---

### 2️⃣ Data Cleaning (`data_cleaning.py`)
**Purpose**: Clean and split data into train/test sets

**Usage**:
```bash
# Default (80/20 split)
python code/data_cleaning.py

# Custom test ratio
python code/data_cleaning.py --test-ratio 0.25

# Custom random seed
python code/data_cleaning.py --random-seed 123
```

**What it does**:
- Cleans match data (handles missing values, outliers)
- Splits into train/test sets
- Saves to `data/train/` and `data/test/`
- Preserves temporal ordering

---

### 3️⃣ Feature Engineering (`feature_engineering.py`)
**Purpose**: Analyze feature importance using Random Forest or SHAP

**Usage**:
```bash
# Random Forest importance
python code/feature_engineering.py --mode rf --top-n 20

# SHAP values (more accurate but slower)
python code/feature_engineering.py --mode shap --sample-size 1000 --top-n 15

# Custom Random Forest
python code/feature_engineering.py --mode rf --n-estimators 500
```

**Parameters**:
- `--mode`: `rf` (Random Forest) or `shap` (SHAP values)
- `--n-estimators`: Number of trees for RF (default: 200)
- `--sample-size`: Samples for SHAP (default: 500)
- `--top-n`: Top N features to display (default: 20)

**Outputs**:
- `results/feature_importance/feature_importance_{mode}.csv`
- `results/feature_importance/feature_importance_{mode}.png`
- `results/feature_importance/shap_summary.png` (SHAP only)

---

### 4️⃣ Model Training (`train.py`)
**Purpose**: Train and evaluate models with hyperparameter tuning

**Usage Examples**:

```bash
# 1. Evaluate all base models (quick baseline)
python code/train.py --evaluate-all

# 2. Train specific models with tuning
python code/train.py --models random_forest xgboost --tune

# 3. Full pipeline (tune + calibrate + save)
python code/train.py --models random_forest gradient_boosting xgboost \
                     --tune --calibrate

# 4. Custom preprocessing
python code/train.py --models logistic_regression --scaler robust --tune
```

**Parameters**:
- `--evaluate-all`: Evaluate all 10 base models
- `--models`: Specific models to train (see Available Models below)
- `--tune`: Enable Bayesian hyperparameter tuning
- `--calibrate`: Apply probability calibration
- `--scaler`: Scaler type (`standard`, `robust`, `minmax`)

**Available Models**:
- `logistic_regression`
- `ridge`
- `random_forest`
- `extra_trees`
- `gradient_boosting`
- `xgboost`
- `adaboost`
- `svc`
- `mlp`
- `knn`

**Outputs**:
- `results/models/{model_name}_model.pkl`
- `results/metrics/base_evaluation.csv`
- `results/metrics/{model_name}_detailed_metrics.csv`
- `results/plots/{model_name}_confusion_matrix.png`
- `results/plots/{model_name}_calibration_curve.png`

---

## 📁 Results Directory Structure

After running the full pipeline:

```
results/
├── feature_importance/
│   ├── feature_importance_rf.csv
│   ├── feature_importance_rf.png
│   ├── feature_importance_shap.csv
│   └── shap_summary.png
├── metrics/
│   ├── base_evaluation.csv
│   ├── random_forest_detailed_metrics.csv
│   └── xgboost_detailed_metrics.csv
├── models/
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
└── plots/
    ├── random_forest_confusion_matrix.png
    ├── random_forest_calibration_curve.png
    └── xgboost_confusion_matrix.png
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Baseline
```bash
# Get baseline performance of all models
python code/train.py --evaluate-all
# Check: results/metrics/base_evaluation.csv
```

### Workflow 2: Production Model
```bash
# 1. Analyze features
python code/feature_engineering.py --mode shap --sample-size 2000

# 2. Train best models with tuning
python code/train.py --models random_forest xgboost gradient_boosting \
                     --tune --calibrate

# 3. Compare results in results/metrics/
```

### Workflow 3: Custom Experiment
```bash
# Try MLP with robust scaling
python code/train.py --models mlp --scaler robust --tune --calibrate
```

---

## ⚙️ Configuration

Edit [code/config.py](code/config.py) to customize:

```python
# Paths
TRAIN_DATA_PATH = 'data/train/full.csv'
TEST_DATA_PATH = 'data/test/full.csv'

# Random seed
RANDOM_SEED = 42

# Hyperparameter grids
PARAM_GRIDS = {
    'random_forest': {...},
    'xgboost': {...},
    ...
}

# Bayesian search settings
BAYESIAN_SEARCH_PARAMS = {
    'n_iter': 50,
    'cv': 5,
    'n_jobs': -1
}
```

---

## 📈 Metrics Explained

### Primary Metric: RPS (Ranked Probability Score)
- **Range**: 0 (perfect) to 1 (worst)
- **Formula**: Sum of squared differences in cumulative probabilities
- **Why**: Handles ordinal outcomes (Home > Draw > Away)

### Secondary Metrics:
- **Accuracy**: Correct predictions / Total predictions
- **Brier Score**: MSE of predicted probabilities (lower is better)
- **Log Loss**: Penalizes confident wrong predictions
- **ROC AUC (OvR)**: Area under ROC curve (One-vs-Rest)

### How to Interpret Results:
1. **Low RPS** (< 0.20): Excellent probability estimates
2. **High Accuracy** (> 0.55): Good categorical predictions
3. **Low Brier** (< 0.25): Well-calibrated probabilities

---

## 🐛 Troubleshooting

### Issue 1: Missing Data
```bash
# Check if data exists
ls data/match/*.csv
ls data/train/full.csv

# If missing, run integration and cleaning
python code/data_integration.py
python code/data_cleaning.py
```

### Issue 2: Memory Error (SHAP)
```bash
# Reduce sample size
python code/feature_engineering.py --mode shap --sample-size 200
```

### Issue 3: Slow Training
```bash
# Reduce Bayesian iterations in config.py
BAYESIAN_SEARCH_PARAMS['n_iter'] = 20  # Instead of 50
```

### Issue 4: Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## 🔬 Advanced Usage

### Custom Feature Engineering
Edit [code/preprocessor.py](code/preprocessor.py):
```python
def create_custom_features(self, df):
    # Add your features here
    df['my_feature'] = ...
    return df
```

### Custom Model
Edit [code/models.py](code/models.py):
```python
def get_base_classifiers():
    classifiers['my_model'] = MyCustomClassifier()
    return classifiers
```

### Custom Metric
Edit [code/metrics.py](code/metrics.py):
```python
def my_custom_metric(y_true, y_pred_proba):
    # Your metric logic
    return score
```

---

## 📚 Next Steps

1. **Ensemble Methods**: Combine top models
   ```python
   from sklearn.ensemble import VotingClassifier
   ```

2. **Deep Learning**: Try LSTM for time series
   ```python
   # See recommendations in summarize.md
   ```

3. **Real-time Pipeline**: Build API for predictions
   ```bash
   # Use FastAPI (see requirements.txt)
   ```

4. **Dashboard**: Visualize predictions
   ```bash
   streamlit run dashboard.py
   ```

---

## 🆘 Need Help?

1. Check [README.md](README.md) for project overview
2. Review [summarize.md](summarize.md) for file descriptions
3. Inspect [code/config.py](code/config.py) for settings
4. Read module docstrings: `python -c "import code.trainer; help(code.trainer)"`

---

**Happy Predicting! ⚽📊**
