# EPL In-Game Prediction - Data Pipeline

## Complete Workflow (Correct Order)

### Step 1: Data Integration
**Purpose**: Merge raw event data with ELO ratings and create match-level features

```bash
python code/data_integration.py --project-dir . --overwrite
```

**Input:**
- `event_data/{matchid}.csv` - Raw events from WhoScored
- `pregame_data/pregame_data.csv` - ELO ratings from ClubElo

**Output:**
- `match/{matchid}.csv` - Full feature set (~27 features per match)
- `data/full.csv` - All matches concatenated

**Features Created:**
- ELO ratings: `ht_elo`, `at_elo`
- Goals: `ht_goal`, `at_goal`
- Passes: `pass`, `short_pass`, `long_pass`, `final_3rd_pass`, `key_pass`
- Shots: `shot`, `shot_6_yard_box`, `shot_penalty_box`, `shot_open_play`, `shot_fast_break`, `big_chance`
- Crosses/Corners: `cross`, `corner`
- Possession: `dispossessed`, `turnover`
- Defense: `duel`, `tackle`, `interception`, `clearance`
- Discipline: `offside`, `yellow`, `red`
- Metadata: `minute`, `half`, `result`

---

### Step 2: Feature Engineering (Importance Analysis)
**Purpose**: Identify top-K most important features using RF or SHAP

```bash
# Option A: Random Forest (fast, ~2-5 minutes)
conda activate rapids-25.10
python code/feature_engineering.py --mode rf --data full --n-estimators 500 --top-n 30

# Option B: SHAP (more accurate, ~10-20 minutes)
conda activate rapids-25.10
python code/feature_engineering.py --mode shap --data full --sample-size 2000 --top-n 30
```

**Input:**
- `data/full.csv` - All matches with full feature set

**Output:**
- `results/feature_importance/rf_importance.csv` - RF-based feature rankings
- `results/feature_importance/shap_importance_class*.csv` - SHAP-based rankings per class
- `results/figures/rf_importance.png` - Visualization
- `results/figures/shap_summary_*.png` - SHAP plots

**What to Check:**
- Open `rf_importance.csv` or `shap_importance_class0.csv`
- Note the top-K feature names (e.g., `ht_goal`, `at_goal`, `ht_elo`, etc.)

---

### Step 3: Data Cleaning with Top-K Selection
**Purpose**: Split data into train/test and keep ONLY top-K features

```bash
# Use top-20 features from SHAP (prefers SHAP, falls back to RF)
python code/data_cleaning.py --test-ratio 0.2 --random-seed 42 \
  --importance-source auto --top-k 20

# Or force RF
python code/data_cleaning.py --test-ratio 0.2 --importance-source rf --top-k 20

# Or disable selection (use all features after drop_features)
python code/data_cleaning.py --test-ratio 0.2 --importance-source none
```

**Input:**
- `match/{matchid}.csv` - Full feature set from integration
- `results/feature_importance/shap_importance_class*.csv` or `rf_importance.csv`

**Output:**
- `train/match/{matchid}.csv` - Top-K features only (e.g., 20 features + result)
- `test/match/{matchid}.csv` - Same top-K features
- `train/data.csv` - All train matches concatenated
- `test/data.csv` - All test matches concatenated

**What to Check:**
```bash
# Verify feature count (should be K + 1 for result)
head -1 data/train/data.csv | tr ',' '\n' | wc -l
# Should output: 21 (if top-k=20)

# List features
head -1 data/train/data.csv
```

**Critical Flags:**
- `--importance-source auto` - Uses SHAP if available, else RF
- `--top-k 20` - Keeps only top-20 features
- If these are omitted or `--top-k 0`, it uses hardcoded `drop_features` list

---

### Step 4: Model Training
**Purpose**: Train models on the filtered dataset

```bash
# Baseline evaluation
python code/train.py --data train --evaluate-all

# Train with tuning (dataset already filtered, so disable --shap-top-k)
python code/train.py --data train \
  --models random_forest xgboost gradient_boosting \
  --tune --calibrate \
  --shap-top-k 0 \
  --wandb --wandb-project epl-in-game-prediction

# Or let training do additional filtering
python code/train.py --data train \
  --models random_forest xgboost \
  --tune --calibrate \
  --shap-top-k 15 --importance-source auto
```

**Input:**
- `train/data.csv` - Already filtered to top-K features

**Output:**
- `results/models/{model}.pkl` - Trained models
- `results/metrics/base_classifier_evaluation.csv` - Baseline metrics
- `results/metrics/{model}_detailed_metrics.csv` - Per-model metrics
- W&B dashboard (if --wandb enabled)

---

## Common Issues & Solutions

### Issue 1: `train/match/{matchid}.csv` has 10-13 features instead of K
**Cause:** You ran `data_cleaning.py` without `--importance-source` and `--top-k` flags

**Solution:**
```bash
# Re-run cleaning with proper flags
python code/data_cleaning.py --test-ratio 0.2 --importance-source auto --top-k 20
```

### Issue 2: "No importance files found"
**Cause:** Feature engineering hasn't been run yet

**Solution:**
```bash
# Run feature engineering first
python code/feature_engineering.py --mode rf --data full --top-n 30
# Then re-run cleaning
python code/data_cleaning.py --test-ratio 0.2 --importance-source auto --top-k 20
```

### Issue 3: Feature mismatch between importance files and match data
**Cause:** You ran feature engineering on cleaned data instead of full data

**Solution:**
```bash
# Re-run feature engineering on full.csv
python code/feature_engineering.py --mode shap --data full --sample-size 2000
```

### Issue 4: "Feature count mismatch"
**Cause:** Feature names in importance files don't match column names in match CSV

**Solution:**
- Check feature names: `head -1 data/match/1284741.csv`
- Check importance file: `head results/feature_importance/rf_importance.csv`
- Ensure they match (case-sensitive)

---

## Quick Verification Commands

```bash
# Check integration output
ls data/match/*.csv | wc -l  # Should show ~380 matches
head -1 data/match/1284741.csv | tr ',' '\n' | wc -l  # Should show ~27 features

# Check feature importance exists
ls results/feature_importance/

# Check cleaned output
head -1 data/train/data.csv | tr ',' '\n' | wc -l  # Should match top-k + 1

# Check final features
head -1 data/train/data.csv
```

---

## Full Pipeline (Copy-Paste)

```bash
# Activate environment
conda activate rapids-25.10

# Step 1: Integration
python code/data_integration.py --project-dir . --overwrite

# Step 2: Feature Engineering
python code/feature_engineering.py --mode rf --data full --n-estimators 500 --top-n 30

# Optional: SHAP analysis (more accurate but slower)
python code/feature_engineering.py --mode shap --data full --sample-size 2000 --top-n 30

# Step 3: Cleaning with top-20 selection
python code/data_cleaning.py --test-ratio 0.2 --random-seed 42 \
  --importance-source auto --top-k 20

# Verify
head -1 data/train/data.csv | tr ',' '\n' | wc -l  # Should be 21

# Step 4: Train
python code/train.py --data train \
  --models random_forest xgboost gradient_boosting \
  --tune --calibrate \
  --shap-top-k 0 \
  --wandb --wandb-project epl-in-game-prediction
```

---

## Pipeline Flowchart

```
event_data/*.csv ──┐
                   ├──> data_integration.py ──> match/*.csv ──> data/full.csv
pregame_data.csv ──┘                                |
                                                    v
                                          feature_engineering.py
                                                    |
                                                    v
                               results/feature_importance/[rf|shap]_importance.csv
                                                    |
                                                    v
                    match/*.csv ──> data_cleaning.py (with --top-k) ──┬──> train/match/*.csv
                                                                       └──> test/match/*.csv
                                                    |
                                                    v
                                            train/data.csv, test/data.csv
                                                    |
                                                    v
                                               train.py ──> results/models/*.pkl
```

---

## Notes

- **Always run feature engineering on `data/full.csv`** (full feature set)
- **Use `--data full` in feature_engineering.py**
- **Use `--importance-source auto --top-k 20` in data_cleaning.py**
- **Set `--shap-top-k 0` in train.py** if data is already filtered (avoid double-filtering)
- **Re-run all steps if you change top-K** to ensure consistency
