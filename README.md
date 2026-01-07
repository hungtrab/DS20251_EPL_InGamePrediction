# EPL In-Game Prediction 

A comprehensive machine learning system for predicting English Premier League match outcomes during live play. This project scrapes event-level data from WhoScored, engineers temporal features, and trains calibrated models to predict match results at any point during a game.

## 🎯 Project Overview

**Goal**: Predict the final outcome (Home Win/Draw/Away Win) of EPL matches while they are in progress, using real-time event data.

**Key Features**:
- Web scraping of detailed event data from WhoScored
- Chronological train/test splitting to prevent data leakage
- Bayesian hyperparameter optimization
- Isotonic calibration for reliable probability estimates
- Multiple evaluation metrics (RPS, Accuracy, Log Loss, ECE, MCE)
- Interactive Streamlit demo for live predictions

**Models Implemented**:
- Random Forest (Best performer: RPS=0.0106)
- XGBoost
- Decision Tree
- K-Nearest Neighbors

**Best Model Performance** (on test set):
- RPS (Ranked Probability Score): 0.0106 ± 0.0002
- Accuracy: 98.95% ± 0.06%
- Log Loss: 0.0864 ± 0.0010
- ECE (Expected Calibration Error): 0.0310
- MCE (Maximum Calibration Error): 0.0580

## 📁 Project Structure

```
DS20251_EPL_InGamePrediction/
├── code/
│   ├── scraping.py                 # WhoScored web scraping
│   ├── data_intergration.ipynb    # Merge event data into full dataset
│   ├── data_cleaning.ipynb        # Feature engineering & cleaning
│   ├── eda.ipynb                  # Exploratory data analysis
│   ├── modelling.ipynb            # Model training & optimization
│   ├── evaluation.ipynb           # Comprehensive model evaluation
│   └── app.py                     # Streamlit demo application
│
├── data/
│   ├── event_data/                # Raw event CSVs per match
│   ├── full.csv                   # Integrated dataset
│   ├── train/                     # Chronologically split training data
│   └── test/                      # Chronologically split test data
│
├── results/
│   ├── models/                    # Trained & calibrated models (.pkl)
│   ├── feature_importance/        # SHAP & RF importance CSVs
│   ├── metrics/                   # Evaluation results
│   └── visualizations/            # Plots & charts
│
├── requirements.txt               # Python dependencies
├── explain.md                     # Detailed technical documentation
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12.12
- Chrome browser (for web scraping)
- Git LFS (for model files)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd DS20251_EPL_InGamePrediction
```

2. **Install Git LFS and pull model files**
```bash
git lfs install
git lfs pull
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download ChromeDriver** (for Selenium)
```bash
# Linux
wget https://chromedriver.storage.googleapis.com/LATEST_RELEASE
wget https://chromedriver.storage.googleapis.com/<VERSION>/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
sudo mv chromedriver /usr/local/bin/

# Or use package manager
sudo apt-get install chromium-chromedriver  # Ubuntu/Debian
```

## 📊 Data Pipeline

### Step 1: Web Scraping

Scrape event-level data from WhoScored for EPL matches:

```bash
cd code
python scraping.py
```

**What it does**:
- Launches headless Chrome browser
- Navigates to WhoScored match pages
- Extracts minute-by-minute events (goals, shots, cards, etc.)
- Saves individual CSV files per match in `data/event_data/`

**Output**: `data/event_data/1284741.csv`, `1284742.csv`, ...

**Key fields**:
- `minute`: Game minute
- `event_type`: Type of event (goal, yellow card, red card, substitution)
- `team`: Home or Away
- `player_name`, `player_in`, `player_out`: Event participants

### Step 2: Data Integration

Combine all event CSVs into a unified dataset:

```python
# In code/data_intergration.ipynb
import pandas as pd
import glob

# Load all event CSVs
event_files = glob.glob('../data/event_data/*.csv')
data = []

for file in event_files:
    match_id = file.split('/')[-1].replace('.csv', '')
    df = pd.read_csv(file)
    df['match_id'] = match_id
    data.append(df)

# Concatenate and save
full_data = pd.concat(data, ignore_index=True)
full_data.to_csv('../data/full.csv', index=False)
```

**Output**: `data/full.csv` (∼500k rows)

### Step 3: Exploratory Data Analysis

Analyze patterns in the data:

```python
# In code/eda.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/full.csv')

# Event distribution by minute
df['minute'].hist(bins=95)
plt.xlabel('Minute')
plt.ylabel('Number of Events')
plt.title('Event Distribution Over Match Time')
plt.show()

# Goal distribution by team
df[df['event_type'] == 'goal'].groupby('team').size().plot(kind='bar')
plt.title('Goals by Team (Home vs Away)')
plt.show()
```

### Step 4: Feature Engineering & Data Cleaning

Transform raw events into predictive features:

```python
# In code/data_cleaning.ipynb
import pandas as pd
import numpy as np

df = pd.read_csv('../data/full.csv')

# Create cumulative features per match per minute
features = df.groupby(['match_id', 'minute']).agg({
    'home_goals': 'last',
    'away_goals': 'last',
    'home_shots': 'sum',
    'away_shots': 'sum',
    'home_yellow_cards': 'sum',
    'away_yellow_cards': 'sum',
    'home_red_cards': 'sum',
    'away_red_cards': 'sum',
    # ... 60+ additional features
}).reset_index()

# Chronological split (prevents data leakage)
match_list = sorted(df['match_id'].unique())
train_matches = match_list[:int(0.8 * len(match_list))]
test_matches = match_list[int(0.8 * len(match_list)):]

train_df = features[features['match_id'].isin(train_matches)]
test_df = features[features['match_id'].isin(test_matches)]

train_df.to_csv('../data/train/train.csv', index=False)
test_df.to_csv('../data/test/test.csv', index=False)
```

**Key Features Engineered** (60+ total):

**Score State**:
- `home_goals`, `away_goals`
- `goal_difference`, `is_home_leading`, `is_away_leading`, `is_draw`

**Event Counts**:
- `home_shots`, `away_shots`, `home_shots_on_target`, `away_shots_on_target`
- `home_yellow_cards`, `away_yellow_cards`, `home_red_cards`, `away_red_cards`
- `home_corners`, `away_corners`, `home_offsides`, `away_offsides`

**Temporal Features**:
- `minute`: Current game minute
- `time_remaining`: Minutes left to play
- `is_first_half`, `is_second_half`

**Momentum Indicators**:
- `home_shots_last_10min`, `away_shots_last_10min`
- `home_goals_last_15min`, `away_goals_last_15min`

**Derived Metrics**:
- `shot_ratio`, `shot_on_target_ratio`
- `corner_ratio`, `possession_proxy`

**Output**: 
- `data/train/train.csv` (chronologically first 80% of matches)
- `data/test/test.csv` (chronologically last 20% of matches)

### Step 5: Model Training

Train and optimize multiple classifiers:

```python
# In code/modelling.ipynb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from skopt import BayesSearchCV
import pandas as pd
import joblib

# Load data
train = pd.read_csv('../data/train/train.csv')
X_train = train.drop(['match_id', 'final_result'], axis=1)
y_train = train['final_result']

# Define search spaces
rf_search_space = {
    'n_estimators': (100, 500),
    'max_depth': (10, 50),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10)
}

# Bayesian optimization
bayes_search = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    rf_search_space,
    n_iter=50,
    cv=5,
    scoring='neg_log_loss',
    random_state=42
)
bayes_search.fit(X_train, y_train)

# Calibrate probabilities
calibrated_model = CalibratedClassifierCV(
    bayes_search.best_estimator_,
    method='isotonic',
    cv=5
)
calibrated_model.fit(X_train, y_train)

# Save model
joblib.dump(calibrated_model, '../results/models/RandomForest_calibrated.pkl')
```

**Training Details**:
- **Optimization**: Bayesian hyperparameter search (50 iterations)
- **Cross-validation**: 5-fold stratified CV
- **Scoring**: Negative log loss
- **Calibration**: Isotonic regression (preserves ordering, better for non-linear calibration)
- **Feature Selection**: SHAP-based importance (top-K features)

**Saved Artifacts**:
- `results/models/RandomForest_calibrated.pkl`
- `results/models/XGBoost_calibrated.pkl`
- `results/models/DecisionTree_calibrated.pkl`
- `results/models/KNN_calibrated.pkl`
- `results/feature_importance/shap_values.csv`
- `results/feature_importance/rf_importance.csv`

### Step 6: Model Evaluation

Comprehensive evaluation on test set:

```python
# In code/evaluation.ipynb
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from torchmetrics.classification import MulticlassCalibrationError
import torch

# Load test data
test = pd.read_csv('../data/test/test.csv')
X_test = test.drop(['match_id', 'final_result'], axis=1)
y_test = test['final_result']

# Load model
model = joblib.load('../results/models/RandomForest_calibrated.pkl')

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Calculate metrics
def rps_score(y_true, y_proba):
    """Ranked Probability Score for ordered outcomes"""
    rps_sum = 0
    for i, true_label in enumerate(y_true):
        cumulative_pred = np.cumsum(y_proba[i])
        cumulative_true = np.zeros(len(y_proba[i]))
        cumulative_true[true_label:] = 1
        rps_sum += np.sum((cumulative_pred - cumulative_true) ** 2)
    return 1 - (rps_sum / len(y_true))

accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_proba)
rps = rps_score(y_test, y_proba)

# Calibration metrics
ece_metric = MulticlassCalibrationError(num_classes=3, n_bins=15, norm='l1')
mce_metric = MulticlassCalibrationError(num_classes=3, n_bins=15, norm='max')

y_proba_tensor = torch.tensor(y_proba, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

ece = ece_metric(y_proba_tensor, y_test_tensor).item()
mce = mce_metric(y_proba_tensor, y_test_tensor).item()

print(f"Accuracy: {accuracy:.4f}")
print(f"Log Loss: {logloss:.4f}")
print(f"RPS Score: {rps:.4f}")
print(f"ECE: {ece:.4f}")
print(f"MCE: {mce:.4f}")
```

**Evaluation Metrics**:

1. **RPS (Ranked Probability Score)**: Primary metric for ordered outcomes (Home Win < Draw < Away Win)
   - Penalizes predictions based on distance from true outcome
   - Range: [0, 1], higher is better
   
2. **Accuracy**: Standard classification accuracy

3. **Log Loss**: Measures prediction confidence calibration

4. **ECE (Expected Calibration Error)**: Average calibration error across confidence bins
   - Measures reliability of predicted probabilities
   
5. **MCE (Maximum Calibration Error)**: Worst-case calibration error

**Output**:
- `results/metrics/base_classifier_evaluation.csv`
- `results/visualizations/calibration_curve_RandomForest.png`
- `results/visualizations/confusion_matrix_RandomForest.png`

### Step 7: Interactive Demo

Launch the Streamlit web application:

```bash
cd code
streamlit run app.py
```

**Demo Features**:

1. **Model Selection**: Choose from trained models (RF, XGBoost, DT, KNN)

2. **Match Selection**: Pick any EPL match from the test set

3. **Minute Slider**: Slide through game minutes (0-90+)

4. **Live Predictions**: See real-time probability updates for:
   - Home Win
   - Draw
   - Away Win

5. **Match Information**: Auto-fetch team names and dates from WhoScored

6. **Visualization**: Interactive bar chart of outcome probabilities

**How to Use**:
1. Select a model from the sidebar
2. Choose a match ID
3. Move the minute slider
4. Watch probabilities update based on in-game events

**Example**:
- Match 1284850 at minute 65
- Home leading 2-1
- Model predicts: Home Win 78%, Draw 15%, Away Win 7%

## 📈 Performance Summary

| Model | RPS Score | Accuracy | Log Loss | ECE | MCE |
|-------|-----------|----------|----------|-----|-----|
| **Random Forest** | **0.9894** | **98.95%** | **0.0864** | **0.0310** | **0.0580** |
| Decision Tree | 0.9896 | 98.44% | 0.5638 | 0.0313 | - |
| KNN | 0.9895 | 97.89% | 0.1297 | 0.0301 | - |
| XGBoost | 0.9601 | 94.75% | 0.2454 | 0.1181 | - |

**Winner**: Random Forest (best RPS, accuracy, and calibration)

## 🔧 Technical Details

### Preventing Data Leakage

**Chronological Splitting**:
```python
# Sort matches by ID (chronological order)
match_list = sorted(df['match_id'].unique(), key=lambda x: int(x))

# Split by time, not randomly
train_matches = match_list[:int(0.8 * len(match_list))]
test_matches = match_list[int(0.8 * len(match_list)):]
```

**Why it matters**: Random splitting would allow the model to "see the future" - training on later matches and testing on earlier ones. Chronological splitting ensures temporal validity.

### Probability Calibration

**Isotonic Regression**:
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    base_model,
    method='isotonic',
    cv=5
)
```

**Why it matters**: Raw model outputs aren't true probabilities. Calibration ensures predicted 70% actually happens 70% of the time.

### Feature Importance

**SHAP Values**:
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

**Top Features** (by SHAP):
1. `goal_difference`
2. `minute`
3. `home_goals`
4. `away_goals`
5. `home_shots_on_target`
6. `time_remaining`
7. `shot_ratio`
8. `home_yellow_cards`
9. `away_yellow_cards`
10. `corner_ratio`

## 🎓 Usage Examples

### Example 1: Train a New Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import joblib

# Load data
train = pd.read_csv('data/train/train.csv')
X_train = train.drop(['match_id', 'final_result'], axis=1)
y_train = train['final_result']

# Train base model
rf = RandomForestClassifier(n_estimators=300, max_depth=30, random_state=42)
rf.fit(X_train, y_train)

# Calibrate
calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)

# Save
joblib.dump(calibrated, 'results/models/MyModel.pkl')
```

### Example 2: Make Predictions

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('results/models/RandomForest_calibrated.pkl')

# Load test data
test = pd.read_csv('data/test/test.csv')
X_test = test.drop(['match_id', 'final_result'], axis=1)

# Predict probabilities
proba = model.predict_proba(X_test)

# Get prediction for a specific match at minute 60
match_1284850_min60 = X_test[(X_test.index == 1234)]  # Example index
prediction = model.predict_proba(match_1284850_min60)

print(f"Home Win: {prediction[0][0]:.2%}")
print(f"Draw: {prediction[0][1]:.2%}")
print(f"Away Win: {prediction[0][2]:.2%}")
```

### Example 3: Evaluate Custom Model

```python
from sklearn.metrics import accuracy_score, log_loss
import joblib
import pandas as pd

# Load model and test data
model = joblib.load('results/models/MyModel.pkl')
test = pd.read_csv('data/test/test.csv')
X_test = test.drop(['match_id', 'final_result'], axis=1)
y_test = test['final_result']

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Log Loss: {logloss:.4f}")
```

## 📚 Additional Documentation

For detailed technical explanations of each pipeline step, see [explain.md](explain.md).

**Topics covered in explain.md**:
- Data schema specifications
- Feature engineering formulas
- Hyperparameter search spaces
- Calibration theory
- Evaluation metric definitions
- Code architecture

## 🐛 Troubleshooting

### Issue: `KeyError: 118` when loading models

**Solution**: Models are stored using Git LFS. Pull them first:
```bash
git lfs pull
```

### Issue: ChromeDriver errors during scraping

**Solution**: Update ChromeDriver to match your Chrome version:
```bash
google-chrome --version  # Check Chrome version
# Download matching ChromeDriver from https://chromedriver.chromium.org/
```

### Issue: `FutureWarning: response_method` in sklearn

**Solution**: Update to scikit-learn 1.4+:
```bash
pip install --upgrade scikit-learn>=1.4.0
```

### Issue: Streamlit app shows "Match not found"

**Solution**: Ensure test data exists:
```bash
ls data/test/  # Should contain test.csv
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **Additional Features**: Add team ratings, player statistics, weather data
2. **Deep Learning**: Implement LSTM/Transformer models for sequence modeling
3. **Real-time Data**: Integrate live API feeds instead of post-match scraping
4. **Deployment**: Dockerize the application, deploy to cloud
5. **Betting Strategy**: Add Kelly Criterion staking, backtesting framework

## 📄 License

This project is for educational purposes only. Data scraped from WhoScored is subject to their terms of service.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

