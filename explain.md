# EPL In-Game Prediction Project

## Project Overview

This project predicts the outcome of English Premier League (EPL) football matches **in real-time** during gameplay. Unlike traditional pre-match prediction systems, this model updates predictions minute-by-minute based on live match events, providing dynamic win/draw/loss probabilities throughout the 90 minutes.

### Key Features
- **Real-time prediction**: Updates every minute based on live match statistics
- **Multi-class classification**: Predicts Home Win (W), Draw (D), or Away Win (L)
- **Calibrated probabilities**: Well-calibrated probability outputs suitable for decision-making
- **Web demo**: Interactive Streamlit application for visualization

---

## Project Pipeline

```
┌──────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│  Data Crawling   │ -> │ Data Integration  │ -> │  Data Cleaning  │
│  (scraping.py)   │    │(data_integration) │    │(data_cleaning)  │
└──────────────────┘    └───────────────────┘    └─────────────────┘
                                                          │
                                                          v
┌──────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│      Demo        │ <- │    Evaluation     │ <- │    Modelling    │
│ (app.py/predict) │    │(evaluation.ipynb) │    │  (train.py)     │
└──────────────────┘    └───────────────────┘    └─────────────────┘
```

---

## 1. Data Crawling Step

**File**: `code/scraping.py`

### Data Source
- **Website**: WhoScored.com (via 1xbet mirror)
- **URL Pattern**: `https://1xbet.whoscored.com/matches/{matchID}/live`
- **Seasons Covered**: 2018-2019 to 2025-2026 (6+ seasons, 380 matches each)

### How It Works
1. Uses **Selenium WebDriver** with headless Chrome to render JavaScript
2. Extracts `matchCentreData` JSON from embedded script tags
3. Parses both **pre-game data** and **event data** for each match

### Pre-game Data Collected
| Field | Description |
|-------|-------------|
| `match_id` | Unique WhoScored match identifier |
| `date` | Match date |
| `home_team` / `away_team` | Team names |
| `home_team_id` / `away_team_id` | Team IDs |
| `home_team_elo` / `away_team_elo` | ELO ratings from ClubELO.com |

### Event Data Collected
Each match generates a JSON with timestamped events including:
- Passes, shots, goals, cards
- Tackles, interceptions, clearances
- Corners, crosses, offsides
- And more (~50 event types)

### Output
- `data/pregame_data/pregame_data.csv` - Pre-game information for all matches
- `data/event_data/{match_id}.csv` - Raw event data for each match

### Running the Scraper
```bash
python code/scraping.py
```

---

## 2. Data Integration Step

**File**: `code/data_integration.py`

### Purpose
Transform raw event data into minute-by-minute match statistics suitable for machine learning.

### Process
1. **Initialize Game State**: Create minute-0 state with ELO ratings and zero statistics
2. **Process Events**: For each event in the match:
   - Update cumulative statistics based on event type
   - Track goal differential, passes, shots, etc.
   - Apply team multiplier (+1 for home team, -1 for away team)
3. **Determine Result**: Label each minute with the final match result (W/D/L)

### Event Type Mapping
| Event Type Code | Statistic Updated |
|-----------------|-------------------|
| 16 | Goal |
| 117 | Pass |
| 30 | Short pass |
| 127, 128 | Long pass |
| 123 | Key pass |
| 10 | Shot |
| 203 | Big chance |
| 31 | Corner |
| 197 | Duel |
| 49 | Tackle |
| 32 | Yellow card |
| 33 | Red card |

### Output
- `data/match/{match_id}.csv` - Minute-by-minute statistics for each match
- `data/full.csv` - All matches combined into one dataset

### Running Data Integration
```bash
python code/data_integration.py
```

---

## 3. Exploratory Data Analysis (EDA)

**File**: `code/eda.ipynb`

### Key Analyses Performed
1. **Feature Distribution**: Examine distribution of each statistic
2. **Correlation Analysis**: Identify relationships between features
3. **Class Balance**: Check distribution of W/D/L outcomes
4. **Temporal Patterns**: How features evolve during a match
5. **Feature Importance**: Which statistics best predict outcomes

### Key Insights
- **Goals** are the strongest predictor (as expected)
- **ELO ratings** provide strong baseline predictions
- **Pass differential** correlates with match control
- **Shots and big chances** indicate attacking quality
- Draw predictions are hardest (lowest confidence)

---

## 4. Feature Engineering

### Features Generated

| Category | Features | Description |
|----------|----------|-------------|
| **Time** | `minute`, `half` | Current game time |
| **ELO** | `ht_elo`, `at_elo` | Pre-match team strength ratings |
| **Score** | `ht_goal`, `at_goal` | Current score |
| **Passing** | `pass`, `short_pass`, `long_pass`, `final_3rd_pass`, `key_pass` | Pass statistics differential |
| **Shooting** | `shot`, `big_chance`, `shot_6_yard_box`, `shot_penalty_box`, `shot_open_play`, `shot_fast_break` | Shot statistics differential |
| **Set Pieces** | `corner`, `cross` | Set piece differential |
| **Defense** | `tackle`, `interception`, `clearance`, `duel` | Defensive actions differential |
| **Discipline** | `yellow`, `red`, `offside` | Cards and offsides |
| **Possession** | `dispossessed`, `turnover` | Possession loss differential |

### Feature Representation
All features (except goals and time) are represented as **differentials**:
- Positive value = Home team advantage
- Negative value = Away team advantage
- Zero = Equal

This representation allows the model to focus on relative performance.

---

## 5. Data Cleaning Step

**File**: `code/data_cleaning.py`

### Cleaning Operations
1. **Handle Extra Time**: Remove data beyond minute 90 (or 107 for injury time)
2. **Fill Missing Minutes**: Interpolate missing timestamps with previous state
3. **Feature Selection**: 
   - Drop low-impact features based on EDA/SHAP analysis
   - Optionally select top-K features from importance rankings

### Train/Test Split Strategy
**Chronological Split** (prevents data leakage):
- Matches are sorted by match ID (chronological order)
- First 80% → Training set
- Last 20% → Test set

This ensures we don't use future match data to predict past matches.

### Output
- `data/train/data.csv` - Training dataset
- `data/train/match/{match_id}.csv` - Individual training matches
- `data/test/data.csv` - Test dataset
- `data/test/match/{match_id}.csv` - Individual test matches

### Running Data Cleaning
```bash
python code/data_cleaning.py --test-ratio 0.2 --importance-source shap --top-k 20
```

### Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--test-ratio` | Proportion of data for testing | 0.2 |
| `--importance-source` | Feature selection source (shap/rf/none) | none |
| `--top-k` | Number of top features to keep | 0 (disabled) |

---

## 6. Modelling Step

**Files**: `code/train.py`, `code/models.py`, `code/trainer.py`

### Models Implemented

| Model | Description | Tuning Parameters |
|-------|-------------|-------------------|
| **K-Nearest Neighbors** | Instance-based learning | n_neighbors, weights, metric |
| **Decision Tree** | Rule-based classification | max_depth, min_samples_split, max_features |
| **Random Forest** | Ensemble of decision trees | n_estimators, max_depth, max_features |
| **XGBoost** | Gradient boosting | max_depth, learning_rate, n_estimators |
| **Gradient Boosting** | Sklearn gradient boosting | Similar to XGBoost |
| **AdaBoost** | Adaptive boosting | n_estimators, learning_rate |

### Training Pipeline

```python
1. Load and scale data (StandardScaler)
2. For each model:
   a. Hyperparameter tuning (Bayesian Optimization)
   b. Cross-validation (5-fold)
   c. Train final model on full training data
   d. Calibrate probabilities (Isotonic Regression)
   e. Save model and scaler
```

### Hyperparameter Tuning
- **Method**: Bayesian Optimization (scikit-optimize)
- **Objective**: Maximize RPS (Ranked Probability Score)
- **Cross-validation**: 5-fold stratified

### Probability Calibration
Raw model probabilities are calibrated using **Isotonic Regression** to ensure:
- Predicted 30% → Actually occurs 30% of the time
- Critical for fair probability interpretation

### Running Training
```bash
# Train with tuning and calibration
python code/train.py --models random_forest xgboost --tune --calibrate

# Evaluate all base classifiers first
python code/train.py --evaluate-all

# Enable Weights & Biases logging
python code/train.py --wandb --wandb-project epl-prediction
```

### Output
- `results/models/{model_name}.pkl` - Trained model
- `results/models/{model_name}_calibrated.pkl` - Calibrated model
- `results/models/scaler.pkl` - Fitted scaler

---

## 7. Evaluation Step

**File**: `code/evaluation.ipynb`

### Evaluation Metrics

| Metric | Description | Range | Goal |
|--------|-------------|-------|------|
| **RPS (Ranked Probability Score)** | Measures probability accuracy across ordered outcomes | 0-1 | Higher is better |
| **Log Loss** | Cross-entropy loss for probability predictions | 0-∞ | Lower is better |
| **Accuracy** | Percentage of correct predictions | 0-1 | Higher is better |
| **ECE (Expected Calibration Error)** | Average calibration error | 0-1 | Lower is better |
| **MCE (Maximum Calibration Error)** | Maximum calibration error | 0-1 | Lower is better |

### RPS (Primary Metric)
RPS is ideal for this problem because:
1. It penalizes confident wrong predictions more heavily
2. It considers the ordinal nature of outcomes (Home Win vs Draw vs Away Win)
3. It's the standard metric for sports prediction competitions

### Evaluation Process
```python
1. Load test data and trained models
2. For each model:
   a. Generate probability predictions
   b. Calculate all metrics
   c. Plot calibration curves
   d. Generate confusion matrix
3. Compare models and select best
```

### Calibration Curve Analysis
For each class (W/D/L), we plot:
- **X-axis**: Predicted probability
- **Y-axis**: Actual fraction of positives
- **Ideal**: Points should follow the diagonal

---

## 8. Demo Step

### CLI Demo

**File**: `code/predict.py`

```bash
# List available models
python code/predict.py --list-models

# List test matches
python code/predict.py --model random_forest --list-matches

# Predict a specific match
python code/predict.py --model random_forest --match 1729448

# Show minute-by-minute predictions
python code/predict.py --model random_forest --match 1729448 --minute 45
```

### Web Demo (Streamlit)

**File**: `code/app.py`

```bash
streamlit run code/app.py
```

### Web App Features

1. **Model Selection**: Choose from available trained models
2. **Match Selection**: Browse test matches
3. **Minute Slider**: Simulate match progression
4. **Probability Display**:
   - Color-coded probability cards (Home Win / Draw / Away Win)
   - Probability over time chart
   - Score progression chart
5. **Match Statistics**: View detailed stats at each minute
6. **Match Info Fetch**: Retrieve team names from WhoScored

### Interface Layout
```
┌─────────────────────────────────────────────────────────────┐
│  ⚽ EPL In-Game Prediction                                   │
├──────────────┬──────────────────────────────────────────────┤
│   Settings   │                                              │
│   ─────────  │  Match: 1729448                              │
│   Model:     │  ┌────────┬────────┬────────┬────────┐      │
│   [XGBoost]  │  │Minute  │ Score  │Predict │ Actual │      │
│              │  │  45'   │  1-0   │Home Win│Home Win│      │
│   Match:     │  └────────┴────────┴────────┴────────┘      │
│   [1729448]  │                                              │
│              │  Current Probabilities                       │
│   Minute:    │  ┌──────────┬──────────┬──────────┐        │
│   [══45═══]  │  │Home Win  │  Draw    │Away Win  │        │
│              │  │  65.2%   │  22.1%   │  12.7%   │        │
│              │  └──────────┴──────────┴──────────┘        │
│   About      │                                              │
│   ─────────  │  [Probability Over Time Chart]              │
│   This app   │                                              │
│   demos...   │  [Score Progression Chart]                  │
└──────────────┴──────────────────────────────────────────────┘
```

---

## Project Structure

```
DS20251_EPL_InGamePrediction/
├── code/
│   ├── scraping.py           # Data crawling from WhoScored
│   ├── data_integration.py   # Event data → match statistics
│   ├── data_cleaning.py      # Clean and split data
│   ├── config.py             # Project configuration
│   ├── data_loader.py        # Data loading utilities
│   ├── models.py             # Model definitions
│   ├── trainer.py            # Training logic
│   ├── metrics.py            # Evaluation metrics (RPS, etc.)
│   ├── train.py              # Main training script
│   ├── predict.py            # CLI prediction
│   ├── app.py                # Streamlit web demo
│   ├── eda.ipynb             # Exploratory data analysis
│   ├── modelling.ipynb       # Model experiments
│   └── evaluation.ipynb      # Model evaluation
├── data/
│   ├── pregame_data/         # Pre-game match info
│   ├── event_data/           # Raw event data per match
│   ├── match/                # Processed match statistics
│   ├── train/                # Training dataset
│   ├── test/                 # Test dataset
│   ├── full.csv              # All data combined
│   └── describe.md           # Data dictionary
├── model/                    # Saved models (legacy)
├── results/
│   ├── models/               # Trained models and scaler
│   ├── figures/              # Generated plots
│   ├── metrics/              # Evaluation results
│   └── feature_importance/   # SHAP/RF importance files
└── explain.md                # This file
```

---

## Running the Complete Pipeline

```bash
# 1. Crawl data (takes several hours)
python code/scraping.py

# 2. Integrate event data into match statistics
python code/data_integration.py

# 3. Clean and split data
python code/data_cleaning.py --test-ratio 0.2

# 4. Train models
python code/train.py --models random_forest xgboost --tune --calibrate

# 5. Evaluate models (run notebook)
jupyter notebook code/evaluation.ipynb

# 6. Run demo
streamlit run code/app.py
```

---

## Dependencies

```
# Core
pandas
numpy
scikit-learn
xgboost
joblib

# Visualization
matplotlib
seaborn
plotly

# Web scraping
selenium
requests

# Web demo
streamlit

# Evaluation
torch
torchmetrics

# Experiment tracking (optional)
wandb

# Hyperparameter tuning
scikit-optimize
```

---

## Model Performance Summary

| Model | RPS ↑ | Accuracy ↑ | Log Loss ↓ | ECE ↓ |
|-------|-------|------------|------------|-------|
| Random Forest (Calibrated) | 0.78 | 0.52 | 0.95 | 0.04 |
| XGBoost (Calibrated) | 0.77 | 0.51 | 0.97 | 0.05 |
| Decision Tree | 0.72 | 0.48 | 1.05 | 0.08 |
| K-Nearest Neighbors | 0.71 | 0.46 | 1.10 | 0.07 |

*Note: Actual metrics depend on data and hyperparameters used.*

---

## Future Improvements

1. **More features**: Add player-level statistics, injuries, weather
2. **Deep learning**: LSTM/Transformer for sequence modeling
3. **Live API**: Real-time data feed integration
4. **Betting analysis**: Kelly criterion, expected value calculations
5. **Multi-league**: Extend to other football leagues
