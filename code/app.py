"""
Streamlit Web App for EPL In-Game Prediction Demo
Run with: streamlit run code/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import joblib
import json
import re
from datetime import datetime

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR, TEST_DATA_PATH

# Try to import requests for API calls
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import selenium for web scraping (WhoScored requires JS)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="EPL In-Game Prediction",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .home-win { background-color: rgba(0, 128, 0, 0.2); }
    .draw { background-color: rgba(128, 128, 128, 0.2); }
    .away-win { background-color: rgba(255, 0, 0, 0.2); }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_match_info(match_id):
    """
    Fetch match info (team names, date, season) from WhoScored API.
    
    Args:
        match_id: The WhoScored match ID (e.g., '1284741')
    
    Returns:
        Dict with keys: home_team, away_team, date, season, or None if failed
    """
    if not SELENIUM_AVAILABLE:
        return None
    
    url = f'https://1xbet.whoscored.com/matches/{match_id}/live'
    driver = None
    
    try:
        # Setup headless Chrome
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Wait for page to load and find matchCentreData
        import time
        time.sleep(3)  # Give page time to load JS
        
        scripts = driver.find_elements(By.TAG_NAME, 'script')
        data_text = None
        
        for script in scripts:
            script_content = script.get_attribute('innerHTML')
            if script_content and 'matchCentreData' in script_content:
                data_text = script_content
                break
        
        if not data_text:
            return None
        
        # Extract JSON from matchCentreData
        start_index = data_text.find('matchCentreData')
        if start_index == -1:
            return None
        
        data_sub = data_text[start_index + len('matchCentreData:'):]
        brace_start = data_sub.find('{')
        if brace_start == -1:
            return None
        
        # Find matching closing brace
        brace_count = 0
        json_end = -1
        for i, char in enumerate(data_sub[brace_start:], start=brace_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end == -1:
            return None
        
        json_str = data_sub[brace_start:json_end]
        data = json.loads(json_str)
        
        # Extract match info
        home_team = data.get('home', {}).get('name', 'Home Team')
        away_team = data.get('away', {}).get('name', 'Away Team')
        start_time = data.get('startTime', '')
        
        # Parse date
        match_date = None
        if start_time:
            try:
                dt = datetime.strptime(start_time.split('T')[0], '%Y-%m-%d')
                match_date = dt.strftime('%B %d, %Y')  # e.g., "January 5, 2026"
            except:
                match_date = start_time.split('T')[0]
        
        # Determine season from date
        season = None
        if start_time:
            try:
                dt = datetime.strptime(start_time.split('T')[0], '%Y-%m-%d')
                year = dt.year
                month = dt.month
                # EPL season runs Aug-May, so if month >= 8, it's the start of a new season
                if month >= 8:
                    season = f"{year}/{year+1}"
                else:
                    season = f"{year-1}/{year}"
            except:
                pass
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'date': match_date,
            'season': season
        }
        
    except Exception as e:
        st.warning(f"Failed to fetch match info: {e}")
        return None
    finally:
        if driver:
            driver.quit()


@st.cache_resource
def load_model(model_name, calibrated=True):
    """Load trained model"""
    suffix = '_calibrated' if calibrated else ''
    model_path = os.path.join(RESULTS_DIR, 'models', f'{model_name}{suffix}.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_resource
def load_scaler():
    """Load scaler"""
    scaler_path = os.path.join(RESULTS_DIR, 'models', 'scaler.pkl')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None


def get_available_models():
    """Get list of available models"""
    models_dir = os.path.join(RESULTS_DIR, 'models')
    if not os.path.exists(models_dir):
        return []
    models = set()
    for f in os.listdir(models_dir):
        if f.endswith('.pkl') and not f.startswith('scaler'):
            name = f.replace('_calibrated.pkl', '').replace('.pkl', '')
            models.add(name)
    return sorted(models)


def get_available_matches():
    """Get list of available test matches"""
    test_dir = os.path.join(os.path.dirname(RESULTS_DIR), 'data', 'test', 'match')
    if not os.path.exists(test_dir):
        return []
    return sorted([f.replace('.csv', '') for f in os.listdir(test_dir) if f.endswith('.csv')])


def predict_match(model, scaler, match_data):
    """Make predictions for match data"""
    if 'result' in match_data.columns:
        X = match_data.drop('result', axis=1).values
    else:
        X = match_data.values
    
    if scaler is not None:
        X = scaler.transform(X)
    
    return model.predict_proba(X)


def create_probability_plot(predictions_df, match_data):
    """Create interactive probability plot"""
    fig = go.Figure()
    
    # Add probability traces
    fig.add_trace(go.Scatter(
        x=predictions_df['minute'],
        y=predictions_df['home_win_prob'],
        name='Home Win',
        line=dict(color='green', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 128, 0, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=predictions_df['minute'],
        y=predictions_df['home_win_prob'] + predictions_df['draw_prob'],
        name='Draw',
        line=dict(color='gray', width=3),
        fill='tonexty',
        fillcolor='rgba(128, 128, 128, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=predictions_df['minute'],
        y=[1] * len(predictions_df),
        name='Away Win',
        line=dict(color='red', width=3),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ))
    
    # Add goal markers
    if 'ht_goal' in match_data.columns:
        ht_goals = match_data['ht_goal'].diff().fillna(0)
        at_goals = match_data['at_goal'].diff().fillna(0)
        
        for i, (ht_g, at_g, minute) in enumerate(zip(ht_goals, at_goals, predictions_df['minute'])):
            if ht_g > 0:
                fig.add_vline(x=minute, line_dash="dash", line_color="green", annotation_text="⚽ Home")
            if at_g > 0:
                fig.add_vline(x=minute, line_dash="dash", line_color="red", annotation_text="⚽ Away")
    
    # Add half-time line
    fig.add_vline(x=45, line_dash="dot", line_color="black", annotation_text="Half-time")
    
    fig.update_layout(
        title="Win Probability Over Time",
        xaxis_title="Minute",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_score_plot(match_data):
    """Create score progression plot"""
    if 'ht_goal' not in match_data.columns:
        return None
    
    fig = go.Figure()
    
    minutes = match_data['minute'] if 'minute' in match_data.columns else range(len(match_data))
    
    fig.add_trace(go.Scatter(
        x=minutes,
        y=match_data['ht_goal'],
        name='Home Goals',
        line=dict(color='green', width=3),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=minutes,
        y=match_data['at_goal'],
        name='Away Goals',
        line=dict(color='red', width=3),
        mode='lines+markers'
    ))
    
    fig.add_vline(x=45, line_dash="dot", line_color="black")
    
    fig.update_layout(
        title="Score Progression",
        xaxis_title="Minute",
        yaxis_title="Goals",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def main():
    st.title("⚽ EPL In-Game Prediction")
    st.markdown("Predict match outcomes in real-time using machine learning")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Model selection
    available_models = get_available_models()
    if not available_models:
        st.error("No trained models found! Run training first.")
        st.code("python code/train.py --models random_forest xgboost --tune --calibrate")
        return
    
    selected_model = st.sidebar.selectbox("Select Model", available_models, index=0)
    use_calibrated = st.sidebar.checkbox("Use Calibrated Model", value=True)
    
    # Load model
    model = load_model(selected_model, calibrated=use_calibrated)
    scaler = load_scaler()
    
    if model is None:
        st.error(f"Failed to load model: {selected_model}")
        return
    
    st.sidebar.success(f"✅ Model loaded: {selected_model}")
    
    # Match selection
    st.sidebar.header("Match Selection")
    
    available_matches = get_available_matches()
    
    if available_matches:
        selected_match = st.sidebar.selectbox(
            "Select Test Match", 
            available_matches,
            index=0
        )
        
        # Load match data
        match_path = os.path.join(os.path.dirname(RESULTS_DIR), 'data', 'test', 'match', f'{selected_match}.csv')
        match_data = pd.read_csv(match_path)
        
        # Get true result
        true_result = None
        result_display = {'W': 'Home Win', 'D': 'Draw', 'L': 'Away Win'}
        if 'result' in match_data.columns:
            true_result = match_data['result'].iloc[0]
        
        # Minute slider for simulation
        max_minute = int(match_data['minute'].max()) if 'minute' in match_data.columns else len(match_data)
        
        st.sidebar.header("Match Simulation")
        current_minute = st.sidebar.slider("Current Minute", 0, max_minute, max_minute)
        
        # Filter data up to current minute
        if 'minute' in match_data.columns:
            current_data = match_data[match_data['minute'] <= current_minute]
        else:
            current_data = match_data.iloc[:current_minute+1]
        
        if len(current_data) == 0:
            st.warning("No data for selected minute")
            return
        
        # Make predictions
        proba = predict_match(model, scaler, current_data)
        
        # Create predictions DataFrame
        minutes = current_data['minute'].values if 'minute' in current_data.columns else np.arange(len(current_data))
        predictions_df = pd.DataFrame({
            'minute': minutes,
            'home_win_prob': proba[:, 0],
            'draw_prob': proba[:, 1],
            'away_win_prob': proba[:, 2]
        })
        
        # Display current state
        st.header(f"Match: {selected_match}")
        
        # Button to manually fetch match info from WhoScored
        info_col1, info_col2 = st.columns([1, 4])
        with info_col1:
            fetch_info = st.button("🔍 Fetch Match Info", help="Fetch team names, date, and season from WhoScored")
        
        # Check if we have cached match info in session state
        if 'match_info_cache' not in st.session_state:
            st.session_state.match_info_cache = {}
        
        # Fetch match info if button clicked
        if fetch_info:
            with st.spinner("Fetching match info from WhoScored..."):
                match_info = fetch_match_info(selected_match)
                if match_info:
                    st.session_state.match_info_cache[selected_match] = match_info
                else:
                    st.warning("Failed to fetch match info. Make sure Selenium is installed.")
        
        # Display match info if available
        if selected_match in st.session_state.match_info_cache:
            match_info = st.session_state.match_info_cache[selected_match]
            with info_col2:
                st.markdown(f"""
                **{match_info['home_team']}** vs **{match_info['away_team']}**  
                📅 {match_info['date'] or 'Unknown date'} | 🏆 EPL {match_info['season'] or 'Unknown season'}
                """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_probs = proba[-1]
        
        with col1:
            st.metric("⏱️ Minute", f"{current_minute}'")
        
        with col2:
            ht_goals = int(current_data['ht_goal'].iloc[-1]) if 'ht_goal' in current_data.columns else 0
            at_goals = int(current_data['at_goal'].iloc[-1]) if 'at_goal' in current_data.columns else 0
            st.metric("📊 Score", f"{ht_goals} - {at_goals}")
        
        with col3:
            predicted = ['Home Win', 'Draw', 'Away Win'][np.argmax(current_probs)]
            st.metric("🎯 Prediction", predicted)
        
        with col4:
            if true_result:
                st.metric("✅ Actual Result", result_display.get(true_result, true_result))
        
        # Probability cards
        st.subheader("Current Probabilities")
        
        prob_col1, prob_col2, prob_col3 = st.columns(3)
        
        with prob_col1:
            st.markdown(f"""
            <div class="result-box home-win">
                <h2 style="color: green;">🏠 Home Win</h2>
                <p class="big-font">{current_probs[0]:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with prob_col2:
            st.markdown(f"""
            <div class="result-box draw">
                <h2 style="color: gray;">🤝 Draw</h2>
                <p class="big-font">{current_probs[1]:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with prob_col3:
            st.markdown(f"""
            <div class="result-box away-win">
                <h2 style="color: red;">✈️ Away Win</h2>
                <p class="big-font">{current_probs[2]:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability plot
        st.subheader("Win Probability Over Time")
        prob_fig = create_probability_plot(predictions_df, current_data)
        st.plotly_chart(prob_fig, use_container_width=True)
        
        # Score plot
        score_fig = create_score_plot(current_data)
        if score_fig:
            st.plotly_chart(score_fig, use_container_width=True)
        
        # Match statistics
        with st.expander("📊 Match Statistics"):
            stats_cols = ['shot', 'big_chance', 'pass', 'key_pass', 'corner']
            available_stats = [c for c in stats_cols if c in current_data.columns]
            
            if available_stats:
                latest = current_data.iloc[-1]
                stats_df = pd.DataFrame({
                    'Statistic': available_stats,
                    'Value': [latest[c] for c in available_stats]
                })
                st.dataframe(stats_df, use_container_width=True)
        
        # Raw data
        with st.expander("📋 Raw Predictions Data"):
            st.dataframe(predictions_df, use_container_width=True)
    
    else:
        st.warning("No test matches found. Run data cleaning first:")
        st.code("python code/data_cleaning.py --test-ratio 0.2 --importance-source shap --top-k 20")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app demonstrates real-time EPL match prediction using machine learning. "
        "Use the minute slider to simulate how predictions change during a match."
    )


if __name__ == "__main__":
    main()
