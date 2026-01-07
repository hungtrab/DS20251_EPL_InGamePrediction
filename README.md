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

```bash
python code/data_integration.py --overwrite
```

**Options**:
- `--project-dir`: Project directory (auto-detected if not specified)
- `--overwrite`: Overwrite existing match files

**Output**: `data/full.csv` (∼500k rows)

### Step 3: Exploratory Data Analysis

Analyze patterns in the data using the Jupyter notebook:

```bash
jupyter notebook code/eda.ipynb
```

**What it covers**:
- Event distribution by minute
- Goal distribution by team (Home vs Away)
- Feature correlation analysis
- Outcome class distribution

### Step 4: Feature Engineering & Data Cleaning

Transform raw events into predictive features and split data:

```bash
python code/data_cleaning.py --test-ratio 0.2 --importance-source auto --top-k 20
```

**Options**:
- `--test-ratio`: Test set ratio (default: 0.2)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--importance-source`: Feature selection source - `none`, `auto`, `shap`, or `rf`
- `--top-k`: Number of top features to keep (0 = disable)

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

```bash
# Train with hyperparameter tuning and calibration
python code/train.py --models random_forest xgboost --tune --calibrate

# Evaluate all base classifiers first
python code/train.py --evaluate-all

# Train with Weights & Biases logging
python code/train.py --models random_forest --tune --wandb --wandb-project epl-prediction
```

**Options**:
- `--models`: Models to train (space-separated): `random_forest`, `xgboost`, `decision_tree`, `knn`
- `--tune`: Perform Bayesian hyperparameter tuning
- `--calibrate`: Apply isotonic calibration (default: True)
- `--evaluate-all`: Evaluate all base classifiers first
- `--scaler`: Scaler type - `standard`, `robust`, or `minmax`
- `--wandb`: Enable Weights & Biases logging
- `--wandb-project`: W&B project name

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

Comprehensive evaluation on test set using the Jupyter notebook:

```bash
jupyter notebook code/evaluation.ipynb
```

Or run evaluation during training:

```bash
python code/train.py --evaluate-all
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
```bash
# Data cleaning automatically uses chronological split
python code/data_cleaning.py --test-ratio 0.2
```

Matches are sorted by ID (chronological order) and split by time, not randomly.

**Why it matters**: Random splitting would allow the model to "see the future" - training on later matches and testing on earlier ones. Chronological splitting ensures temporal validity.

### Probability Calibration

**Isotonic Regression**:
```bash
# Calibration is enabled by default during training
python code/train.py --models random_forest --calibrate
```

**Why it matters**: Raw model outputs aren't true probabilities. Calibration ensures predicted 70% actually happens 70% of the time.

### Feature Importance

**SHAP Values**:
```bash
# Feature selection uses SHAP importance during data cleaning
python code/data_cleaning.py --importance-source shap --top-k 20
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

### Example 1: Full Pipeline

```bash
# Step 1: Integrate event data
python code/data_integration.py --overwrite

# Step 2: Clean and split data
python code/data_cleaning.py --test-ratio 0.2 --importance-source auto --top-k 20

# Step 3: Train models with tuning
python code/train.py --models random_forest xgboost --tune --calibrate

# Step 4: Launch demo app
streamlit run code/app.py
```

### Example 2: Train Specific Model

```bash
# Train only Random Forest with hyperparameter tuning
python code/train.py --models random_forest --tune --calibrate

# Train multiple models without tuning
python code/train.py --models random_forest xgboost decision_tree knn
```

### Example 3: Evaluate All Models

```bash
# Evaluate all base classifiers
python code/train.py --evaluate-all

# Or use the evaluation notebook for detailed analysis
jupyter notebook code/evaluation.ipynb
```

### Example 4: Train with Experiment Tracking

```bash
# Train with Weights & Biases logging
python code/train.py --models random_forest --tune --wandb --wandb-project epl-prediction --wandb-run-name rf-tuned
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

