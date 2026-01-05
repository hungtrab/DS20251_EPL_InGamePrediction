# EPL In-Game Prediction Demo Guide

This guide explains how to run demos for the EPL In-Game Prediction project.

## Quick Start

### 1. Command-Line Demo

```bash
# List available trained models
python code/predict.py --list-models

# List available test matches
python code/predict.py --model random_forest --list-matches

# Run prediction on a specific match
python code/predict.py --model random_forest --match-id 1284743

# Run without calibration
python code/predict.py --model xgboost --match-id 1284743 --no-calibrated

# Save plot without displaying
python code/predict.py --model random_forest --match-id 1284743 --no-plot
```

### 2. Interactive Web App

```bash
# Install Streamlit and Plotly if not installed
pip install streamlit plotly

# Run the Streamlit app
streamlit run code/app.py
```

The web app will open at `http://localhost:8501`

---

## Demo Features

### Command-Line Tool (`predict.py`)

- **Minute-by-minute predictions**: Shows probability changes throughout the match
- **Visualization**: Generates probability area chart with goal markers
- **Multiple models**: Compare predictions from different models
- **Calibration toggle**: Use calibrated or uncalibrated probabilities

### Web App (`app.py`)

- **Interactive minute slider**: Simulate real-time predictions
- **Model comparison**: Switch between different trained models
- **Live probability updates**: Watch predictions change as match progresses
- **Match statistics**: View key stats like shots, passes, corners
- **Score progression**: Track goal events

---

## Demo Workflow

### Before the Demo

1. **Ensure models are trained**:
   ```bash
   # Check for trained models
   ls model/models/*.pkl
   
   # If no models, run training
   python code/train.py --models random_forest xgboost --tune --calibrate
   ```

2. **Prepare test data**:
   ```bash
   # Check for test matches
   ls data/test/match/
   
   # If no test data, run data cleaning
   python code/data_cleaning.py --test-ratio 0.2 --importance-source shap --top-k 20
   ```

### During the Demo

1. **Start with the web app** - More visually appealing for presentations
2. **Select a match** - Pick one with interesting goal patterns
3. **Use the minute slider** - Show how predictions evolve
4. **Highlight key moments**:
   - How probability shifts after goals
   - Model confidence changes over time
   - Comparison to actual result

### Presentation Talking Points

1. **Project Overview**
   - Predicting EPL match outcomes in real-time
   - Using minute-by-minute match statistics
   - Multiple ML models with hyperparameter tuning

2. **Technical Highlights**
   - Feature importance analysis (SHAP values)
   - Probability calibration for better estimates
   - Bayesian hyperparameter optimization

3. **Model Performance**
   - RPS (Ranked Probability Score) metric
   - Calibration curves
   - Feature importance visualization

---

## Sample Demo Commands

### Show Model Comparison

```bash
# Predict with different models on the same match
python code/predict.py --model random_forest --match-id 1284750
python code/predict.py --model xgboost --match-id 1284750
python code/predict.py --model logistic_regression --match-id 1284750
```

### Show Calibration Effect

```bash
# Compare calibrated vs uncalibrated
python code/predict.py --model random_forest --match-id 1284750
python code/predict.py --model random_forest --match-id 1284750 --no-calibrated
```

---

## Troubleshooting

### "No models found"
```bash
python code/train.py --models random_forest xgboost --calibrate
```

### "No test matches found"
```bash
python code/data_cleaning.py --test-ratio 0.2
```

### Streamlit not installed
```bash
pip install streamlit plotly
```

### Port already in use
```bash
streamlit run code/app.py --server.port 8502
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `code/predict.py` | CLI prediction tool with visualization |
| `code/app.py` | Interactive Streamlit web app |
| `DEMO.md` | This guide |
| `PIPELINE.md` | Full pipeline documentation |

---

## Screenshots

### Web App Main View
- Match selection dropdown
- Current minute slider
- Live probability display
- Interactive probability chart

### CLI Output
- Match summary with final prediction
- Probability progression table
- Saved visualization plot

---

## Tips for a Great Demo

1. **Choose interesting matches** - Ones with lead changes or late goals
2. **Use the simulation** - Slowly slide through minutes to show evolution
3. **Compare models** - Show how different models react to events
4. **Show actual vs predicted** - Highlight correct and incorrect predictions
5. **Explain probabilities** - Help audience understand what 60% home win means

Good luck with your demo! ⚽
