# Project Summary: EPL In-Game Prediction

This document provides a comprehensive summary of all files in the DS20251_EPL_InGamePrediction project.

## Project Overview
This project focuses on predicting English Premier League (EPL) match outcomes in real-time using machine learning models. It scrapes match data, processes event-level statistics, and builds prediction models to estimate win/draw/loss probabilities as matches progress.

---

## Code Files

### 1. scraping.py
**Purpose:** Web scraping script to collect EPL match data from WhoScored.com

**Key Features:**
- Uses Selenium WebDriver to scrape match data from 1xbet.whoscored.com
- Extracts event-level data and pre-game information (team ELOs, match IDs, dates)
- Fetches team ELO ratings from ClubElo API
- Processes multiple seasons of EPL matches (2018-2025)
- Implements error handling and retry logic
- Saves two types of data:
  - Event data: Individual match events (shots, passes, etc.) saved as CSV per match
  - Pre-game data: Team names, IDs, ELO ratings, match dates saved in a single CSV

**Technical Details:**
- Parses JavaScript data embedded in HTML (`matchCentreData` JSON object)
- Implements team name normalization (e.g., "Man Utd" → "Man United")
- Uses headless Chrome for automated browsing
- Includes polite delay between requests (2-5 seconds)

---

### 2. data_intergration.ipynb
**Purpose:** Integrates pre-game data with event data to create comprehensive match datasets

**Key Process:**
1. Loads pre-game data (team ELOs, IDs, match info)
2. For each match, reads corresponding event data
3. Transforms event data from raw format to structured attributes
4. Creates minute-by-minute snapshots of match statistics
5. Calculates cumulative differences between home and away teams for each metric

**Output Attributes Generated:**
- Time: minute, half
- Team strength: ht_elo, at_elo
- Score: ht_goal, at_goal
- Passing metrics: pass, short_pass, long_pass, final_3rd_pass, key_pass
- Attacking metrics: cross, corner, big_chance, shot (various types)
- Defensive metrics: dispossessed, turnover, duel, tackle, interception, clearance
- Discipline: offside, yellow, red cards
- Result: W/D/L from home team perspective

**Data Structure:**
- Creates one data instance per minute of each match
- Stores individual match files in `data/match/`
- Aggregates all matches into `data/full.csv`

---

### 3. data_cleaning.ipynb
**Purpose:** Cleans and prepares integrated match data for modeling

**Cleaning Steps:**
1. **Data Splitting:** Divides matches into 80% training and 20% test sets
2. **Feature Selection:** Removes low-impact attributes:
   - short_pass, long_pass, cross
   - shot_6_yard_box, shot_penalty_box, shot_open_play, shot_fast_break
   - dispossessed, turnover, duel, tackle, interception, clearance, offside, yellow
3. **Data Quality:**
   - Eliminates error records (matches with half > 2 or minute > 107)
   - Fills missing minute data with previous state values
   - Ensures continuous time series for each match
4. **Output Structure:**
   - Training data: `data/train/data.csv` and `data/train/match/`
   - Test data: `data/test/data.csv` and `data/test/match/`

---

### 4. eda.ipynb
**Purpose:** Exploratory Data Analysis - visualizes and analyzes match statistics

**Key Analyses:**
1. **Data Distribution:**
   - Histogram of all attributes
   - Minute distribution showing extra time patterns
   - Half distribution

2. **Correlation Analysis:**
   - Heatmap of all features vs. result
   - Result distribution (Win/Draw/Loss)
   - Relationship between passes, shots, and goals

3. **Key Insights:**
   - Shot count strongly correlates with goals
   - ELO difference impacts match results
   - Red cards negatively affect goal difference
   - Corner kicks create goal-scoring opportunities
   - Home teams tend to have more possession

4. **Full-Time Analysis (Minute 90):**
   - Distribution of goal differences
   - Shot and pass patterns at match end
   - Win/draw/loss distribution
   - Relationship between key passes and big chances

5. **Temporal Patterns:**
   - Average statistics progression over match time
   - Individual match examples showing event trends

---

### 5. modelling copy.ipynb
**Purpose:** Main modeling notebook for training and hyperparameter tuning

**Workflow:**

**Data Loading & Preprocessing:**
- Loads training data from `data/train/data.csv`
- Feature scaling using StandardScaler
- Label encoding: W→0, D→1, L→2

**Model Selection (Initial):**
Tests 10 classifiers with 10-fold Stratified Shuffle Split:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Gaussian Naive Bayes
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis

**Scoring Metric:**
- Ranked Probability Score (RPS): Custom metric for probabilistic predictions
- Formula penalizes incorrect probability distributions

**Hyperparameter Tuning:**
Uses Bayesian Optimization (BayesSearchCV) for top models:

1. **K-Nearest Neighbors:**
   - n_neighbors: 3-20
   - weights: uniform/distance
   - metric: minkowski/euclidean/manhattan

2. **Decision Tree:**
   - max_depth: 3-40
   - min_samples_split: 2-40
   - max_features: None/sqrt/log2
   - class_weight: None/balanced

3. **Random Forest:**
   - n_estimators: 100-1000
   - max_depth: 10-100
   - min_samples_split: 2-20
   - bootstrap: True/False
   - Also extracts feature importance

4. **XGBoost:**
   - max_depth: 3-10
   - n_estimators: 100-1000
   - learning_rate: 0.01-0.3
   - colsample_bytree: 0.5-1
   - subsample: 0.6-1

**Model Calibration:**
- Creates calibrated versions using CalibratedClassifierCV
- Method: isotonic regression
- Uses 20% calibration set split from training data

**Outputs:**
- Saves 8 models to `model/` directory:
  - Original tuned models (KNN, Decision Tree, Random Forest, XGBoost)
  - Calibrated versions of each
- Feature importance DataFrame for Random Forest

---

### 6. modelling.ipynb
**Purpose:** Alternative modeling notebook (similar to modelling copy.ipynb but may have different configurations)

**Structure:** Same workflow as modelling copy.ipynb with identical sections for model selection, hyperparameter tuning, and calibration.

---

### 7. modelling_cuml.ipynb
**Purpose:** GPU-accelerated modeling using RAPIDS cuML library

**Key Difference:**
- Uses cuML implementations for GPU acceleration
- Intended for faster training on NVIDIA GPUs
- Same models and hyperparameter search as standard version
- Note: Some cells show errors, likely due to GPU/cuML availability

---

### 8. evaluation.ipynb
**Purpose:** Evaluates all trained models on the test dataset

**Evaluation Process:**

1. **Data Loading:**
   - Loads test data from `data/test/data.csv`
   - Applies same preprocessing (StandardScaler, label encoding)

2. **Evaluation Metrics:**
   - **Ranked Probability Score (RPS):** Primary metric
   - **Expected Calibration Error (ECE):** Measures probability calibration quality
   - **Maximum Calibration Error (MCE):** Worst-case calibration error

3. **Calibration Analysis:**
   - Generates calibration curves for each outcome class (W/D/L)
   - Plots predicted probability vs. true probability
   - Perfect calibration would follow diagonal line

4. **Model Comparison:**
   - Evaluates all models in `model/` directory
   - Creates comparison DataFrame with RPS, ECE, MCE scores
   - Visualizes performance with horizontal bar charts

5. **Demo Model Selection:**
   - Saves best-performing XGBoost model as `demo_model.pkl`

---

### 9. demo.ipynb
**Purpose:** Demonstrates real-time prediction visualization on a sample match

**Demo Features:**
1. Loads the pre-trained demo model (`demo_model.pkl`)
2. Loads a sample match data file (`1729302.csv`)
3. Generates minute-by-minute win probability predictions
4. Visualizes predictions as a stacked area chart showing:
   - Win (W) probability
   - Draw (D) probability
   - Loss (L) probability
   - Over match duration

**Visualization:**
- X-axis: Time (minute by minute)
- Y-axis: Probability percentage
- Three colored areas stacked to show probability distribution evolution
- Includes reference to actual WhoScored match page

---

### 10. debug_page_structure.py
**Purpose:** Debugging script for web scraping development

**Functionality:**
- Tests Selenium setup on a known match ID
- Searches through all `<script>` tags to locate match data
- Identifies tags containing keywords like 'matchCentreData', 'eventData'
- Saves full page source to `debug_page_source.html` for manual inspection
- Helps troubleshoot changes to WhoScored website structure

---

### 11. debug_page_source.html
**Purpose:** Saved HTML output from debug script

**Contents:**
- Full page source from a WhoScored match page
- Used for offline analysis of page structure
- Helps identify where match data is embedded in the page
- Contains CSS styles and JavaScript code (truncated in attachment)

---

## Data Files

### data/describe.md
**Purpose:** Data dictionary documenting all attributes

**Documented Attributes (29 total):**
- **Temporal:** minute, half
- **Team Strength:** ht_elo, at_elo
- **Score:** ht_goal, at_goal
- **Passing:** pass, short_pass, long_pass, final_3rd_pass, key_pass
- **Attacking:** cross, corner, big_chance, shot, shot_6_yard_box, shot_penalty_box, shot_open_play, shot_fast_break
- **Ball Control:** dispossessed, turnover
- **Defensive:** duel, tackle, interception, clearance
- **Discipline:** offside, yellow, red
- **Outcome:** result (W/D/L from home team perspective)

**Note:** Most attributes represent the difference between home team and away team statistics (positive = home team advantage)

---

### data/full.csv
**Purpose:** Complete dataset combining all match data

**Structure:**
- Aggregation of all individual match files
- Minute-by-minute records for all matches
- Used for initial EDA before train/test split
- Contains all 29 attributes defined in describe.md

---

### data/event_data/
**Purpose:** Raw event-level data scraped from WhoScored

**Contents:**
- One CSV file per match (e.g., 1284741.csv, 1284742.csv)
- Match IDs range across multiple seasons
- Contains raw event records with columns like:
  - period, type, satisfiedEventsTypes (as dictionary strings)
  - minute, teamId, other event-specific attributes
- Used as input for data_intergration.ipynb

---

### data/pregame_data/pregame_data.csv
**Purpose:** Pre-match information for all scraped matches

**Columns:**
- match_id: WhoScored match identifier
- date: Match date
- home_team, away_team: Team names
- home_team_id, away_team_id: WhoScored team IDs
- home_team_elo, away_team_elo: ELO ratings from ClubElo API

---

### data/train/ and data/test/
**Purpose:** Cleaned and split datasets for modeling

**Structure:**
- `data.csv`: Aggregated training/test data
- `match/`: Individual match files
- Training set: ~80% of matches
- Test set: ~20% of matches
- Both contain reduced feature set (15 attributes after feature selection)

---

## Model Directory

### model/
**Purpose:** Stores trained machine learning models

**Saved Models:**
- K Nearest Neighbors.pkl / Calibrated K Nearest Neighbors.pkl
- Decision Tree.pkl / Calibrated Decision Tree.pkl
- Random Forest.pkl / Calibrated Random Forest.pkl
- XGBoost.pkl / Calibrated XGBoost.pkl
- demo_model.pkl (best model for demonstration)

**Model Format:** Joblib serialized scikit-learn/XGBoost models

---

## Project Workflow Summary

1. **Data Collection (scraping.py):**
   - Scrape match data from WhoScored.com
   - Fetch ELO ratings from ClubElo API
   - Save event data and pre-game data

2. **Data Integration (data_intergration.ipynb):**
   - Combine pre-game and event data
   - Create minute-by-minute match states
   - Calculate team statistics differences

3. **Data Cleaning (data_cleaning.ipynb):**
   - Remove low-impact features
   - Handle missing minutes
   - Split into train/test sets

4. **Exploratory Analysis (eda.ipynb):**
   - Visualize distributions and correlations
   - Identify key predictive features
   - Understand match dynamics

5. **Model Training (modelling copy.ipynb, modelling.ipynb):**
   - Test multiple classifiers
   - Optimize hyperparameters with Bayesian search
   - Create calibrated probability models
   - Save trained models

6. **Model Evaluation (evaluation.ipynb):**
   - Test on held-out data
   - Calculate RPS, ECE, MCE metrics
   - Generate calibration curves
   - Select best model for deployment

7. **Demonstration (demo.ipynb):**
   - Load trained model
   - Predict match outcomes minute-by-minute
   - Visualize probability evolution

---

## Key Technologies Used

- **Web Scraping:** Selenium, BeautifulSoup
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, RAPIDS cuML
- **Optimization:** Scikit-optimize (Bayesian optimization)
- **Model Evaluation:** TorchMetrics (calibration metrics)
- **Model Persistence:** Joblib

---

## Project Metrics

- **Seasons Covered:** 2018-2019 through 2024-2025 (potential for 2025-2026)
- **Matches per Season:** ~380 EPL matches
- **Data Granularity:** Minute-by-minute (0-107+ minutes per match)
- **Features Used:** 15 (after feature selection from 29)
- **Models Trained:** 10 classifiers, 4 optimized + 4 calibrated variants
- **Primary Metric:** Ranked Probability Score (RPS)
- **Secondary Metrics:** Expected Calibration Error (ECE), Maximum Calibration Error (MCE)

---

## Project Status

✅ **Completed:**
- Data collection pipeline
- Data integration and cleaning
- Exploratory data analysis
- Model training and hyperparameter tuning
- Model evaluation and comparison
- Demonstration notebook

⚠️ **Notes:**
- modelling_cuml.ipynb shows errors (likely GPU/cuML setup issues)
- Debug files remain from development process
- Some season ranges commented out in scraping.py (can be enabled)

---

*End of Summary*
