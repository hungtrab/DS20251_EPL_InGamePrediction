"""
Prediction script for EPL In-Game Prediction
Load trained models and make predictions on match data
"""
import os
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_DIR, TEST_DATA_PATH, RESULT_MAP
from data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Reverse mapping for display
RESULT_NAMES = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}


def load_model(model_name, calibrated=True):
    """Load a trained model from disk"""
    suffix = '_calibrated' if calibrated else ''
    model_path = os.path.join(RESULTS_DIR, 'models', f'{model_name}{suffix}.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logging.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


def load_scaler():
    """Load the fitted scaler"""
    scaler_path = os.path.join(RESULTS_DIR, 'models', 'scaler.pkl')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None


def predict_match(model, scaler, match_data):
    """
    Predict probabilities for a single match state
    
    Args:
        model: Trained classifier
        scaler: Fitted scaler (or None)
        match_data: DataFrame with match features
    
    Returns:
        Probabilities for [Home Win, Draw, Away Win]
    """
    # Drop result column if present
    if 'result' in match_data.columns:
        X = match_data.drop('result', axis=1).values
    else:
        X = match_data.values
    
    # Scale features
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict
    proba = model.predict_proba(X)
    return proba


def predict_match_file(model, scaler, match_file):
    """
    Predict probabilities for all minutes in a match
    
    Args:
        model: Trained classifier
        scaler: Fitted scaler
        match_file: Path to match CSV
    
    Returns:
        DataFrame with minute-by-minute predictions
    """
    logging.info(f"Loading match: {match_file}")
    match_data = pd.read_csv(match_file)
    
    # Get true result if available
    true_result = None
    if 'result' in match_data.columns:
        true_result = match_data['result'].iloc[-1]
        X = match_data.drop('result', axis=1)
    else:
        X = match_data
    
    # Get minute column
    minutes = X['minute'].values if 'minute' in X.columns else np.arange(len(X))
    
    # Scale and predict
    X_values = X.values
    if scaler is not None:
        X_values = scaler.transform(X_values)
    
    proba = model.predict_proba(X_values)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'minute': minutes,
        'home_win_prob': proba[:, 0],
        'draw_prob': proba[:, 1],
        'away_win_prob': proba[:, 2],
        'predicted_result': [RESULT_NAMES[np.argmax(p)] for p in proba]
    })
    
    if true_result is not None:
        results['true_result'] = true_result
    
    return results, match_data


def plot_match_predictions(predictions, match_data, save_path=None, show=True):
    """
    Plot prediction probabilities over match time
    
    Args:
        predictions: DataFrame with predictions
        match_data: Original match data
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    minutes = predictions['minute']
    
    # Plot 1: Probabilities over time
    ax1 = axes[0]
    ax1.plot(minutes, predictions['home_win_prob'], 'g-', linewidth=2, label='Home Win')
    ax1.plot(minutes, predictions['draw_prob'], 'gray', linewidth=2, label='Draw')
    ax1.plot(minutes, predictions['away_win_prob'], 'r-', linewidth=2, label='Away Win')
    ax1.fill_between(minutes, 0, predictions['home_win_prob'], alpha=0.3, color='green')
    ax1.fill_between(minutes, predictions['home_win_prob'], 
                     predictions['home_win_prob'] + predictions['draw_prob'], 
                     alpha=0.3, color='gray')
    ax1.fill_between(minutes, predictions['home_win_prob'] + predictions['draw_prob'], 
                     1, alpha=0.3, color='red')
    
    # Mark goals
    if 'ht_goal' in match_data.columns and 'at_goal' in match_data.columns:
        ht_goals = match_data['ht_goal'].diff().fillna(0)
        at_goals = match_data['at_goal'].diff().fillna(0)
        
        for i, (ht_g, at_g) in enumerate(zip(ht_goals, at_goals)):
            if ht_g > 0:
                ax1.axvline(x=minutes.iloc[i] if hasattr(minutes, 'iloc') else minutes[i], 
                           color='green', linestyle='--', alpha=0.7)
                ax1.annotate('⚽ Home', (minutes.iloc[i] if hasattr(minutes, 'iloc') else minutes[i], 0.95),
                            fontsize=10, ha='center')
            if at_g > 0:
                ax1.axvline(x=minutes.iloc[i] if hasattr(minutes, 'iloc') else minutes[i], 
                           color='red', linestyle='--', alpha=0.7)
                ax1.annotate('⚽ Away', (minutes.iloc[i] if hasattr(minutes, 'iloc') else minutes[i], 0.05),
                            fontsize=10, ha='center')
    
    ax1.set_xlim(0, max(minutes))
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Minute', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('In-Game Win Probability', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add half-time line
    ax1.axvline(x=45, color='black', linestyle=':', alpha=0.5, label='Half-time')
    
    # Plot 2: Score and key stats
    ax2 = axes[1]
    if 'ht_goal' in match_data.columns and 'at_goal' in match_data.columns:
        ax2.plot(minutes, match_data['ht_goal'], 'g-', linewidth=2, marker='o', 
                markersize=3, label='Home Goals')
        ax2.plot(minutes, match_data['at_goal'], 'r-', linewidth=2, marker='o', 
                markersize=3, label='Away Goals')
    
    ax2.set_xlim(0, max(minutes))
    ax2.set_xlabel('Minute', fontsize=12)
    ax2.set_ylabel('Goals', fontsize=12)
    ax2.set_title('Score Progression', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=45, color='black', linestyle=':', alpha=0.5)
    
    # Add final result annotation
    if 'true_result' in predictions.columns:
        true_result = predictions['true_result'].iloc[0]
        result_map_display = {'W': 'Home Win', 'D': 'Draw', 'L': 'Away Win'}
        final_result = result_map_display.get(true_result, true_result)
        
        final_ht = match_data['ht_goal'].iloc[-1] if 'ht_goal' in match_data.columns else '?'
        final_at = match_data['at_goal'].iloc[-1] if 'at_goal' in match_data.columns else '?'
        
        fig.suptitle(f'Final Score: {int(final_ht)} - {int(final_at)} ({final_result})', 
                    fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def demo_single_match(model_name, match_file, calibrated=True, save_plot=True):
    """
    Demo: Predict and visualize a single match
    """
    # Load model and scaler
    model = load_model(model_name, calibrated=calibrated)
    scaler = load_scaler()
    
    # Predict
    predictions, match_data = predict_match_file(model, scaler, match_file)
    
    # Display summary
    print("\n" + "=" * 60)
    print(f"Match Prediction Summary")
    print("=" * 60)
    
    # Final prediction
    final_probs = predictions.iloc[-1]
    print(f"\nFinal Prediction (Minute {int(final_probs['minute'])}):")
    print(f"  Home Win: {final_probs['home_win_prob']:.1%}")
    print(f"  Draw:     {final_probs['draw_prob']:.1%}")
    print(f"  Away Win: {final_probs['away_win_prob']:.1%}")
    print(f"  Predicted: {final_probs['predicted_result']}")
    
    if 'true_result' in predictions.columns:
        result_map_display = {'W': 'Home Win', 'D': 'Draw', 'L': 'Away Win'}
        true = result_map_display.get(predictions['true_result'].iloc[0], predictions['true_result'].iloc[0])
        print(f"  Actual:    {true}")
        
        # Check if correct
        if final_probs['predicted_result'] == true:
            print("  ✅ CORRECT!")
        else:
            print("  ❌ INCORRECT")
    
    # Show key moments
    print("\nKey Moments (probability shifts):")
    prob_diff = predictions[['home_win_prob', 'draw_prob', 'away_win_prob']].diff().abs().sum(axis=1)
    key_moments = predictions[prob_diff > 0.1].head(5)
    for _, row in key_moments.iterrows():
        print(f"  Minute {int(row['minute'])}: Home {row['home_win_prob']:.1%}, "
              f"Draw {row['draw_prob']:.1%}, Away {row['away_win_prob']:.1%}")
    
    # Plot
    if save_plot:
        plot_path = os.path.join(RESULTS_DIR, 'figures', f'match_prediction_{os.path.basename(match_file).replace(".csv", ".png")}')
        plot_match_predictions(predictions, match_data, save_path=plot_path, show=True)
    else:
        plot_match_predictions(predictions, match_data, show=True)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='EPL In-Game Prediction Demo')
    parser.add_argument('--model', type=str, default='random_forest',
                       help='Model to use for prediction')
    parser.add_argument('--match', type=str, default=None,
                       help='Path to match CSV file')
    parser.add_argument('--match-id', type=str, default=None,
                       help='Match ID to predict (looks in test/match/)')
    parser.add_argument('--no-calibrated', action='store_true',
                       help='Use uncalibrated model')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')
    parser.add_argument('--list-models', action='store_true',
                       help='List available trained models')
    parser.add_argument('--list-matches', action='store_true',
                       help='List available test matches')
    
    args = parser.parse_args()
    
    # List available models
    if args.list_models:
        models_dir = os.path.join(RESULTS_DIR, 'models')
        if os.path.exists(models_dir):
            models = [f.replace('.pkl', '') for f in os.listdir(models_dir) 
                     if f.endswith('.pkl') and not f.startswith('scaler')]
            print("Available models:")
            for m in sorted(set(models)):
                print(f"  - {m}")
        return
    
    # List available matches
    if args.list_matches:
        test_match_dir = os.path.join(os.path.dirname(RESULTS_DIR), 'data', 'test', 'match')
        if os.path.exists(test_match_dir):
            matches = [f.replace('.csv', '') for f in os.listdir(test_match_dir) if f.endswith('.csv')]
            print(f"Available test matches ({len(matches)} total):")
            for m in sorted(matches)[:20]:
                print(f"  - {m}")
            if len(matches) > 20:
                print(f"  ... and {len(matches) - 20} more")
        return
    
    # Determine match file
    if args.match:
        match_file = args.match
    elif args.match_id:
        match_file = os.path.join(os.path.dirname(RESULTS_DIR), 'data', 'test', 'match', f'{args.match_id}.csv')
    else:
        # Use first test match as default
        test_match_dir = os.path.join(os.path.dirname(RESULTS_DIR), 'data', 'test', 'match')
        if os.path.exists(test_match_dir):
            matches = [f for f in os.listdir(test_match_dir) if f.endswith('.csv')]
            if matches:
                match_file = os.path.join(test_match_dir, sorted(matches)[0])
                logging.info(f"Using default test match: {match_file}")
            else:
                logging.error("No test matches found")
                return
        else:
            logging.error("Test match directory not found")
            return
    
    if not os.path.exists(match_file):
        logging.error(f"Match file not found: {match_file}")
        return
    
    # Run demo
    demo_single_match(
        args.model, 
        match_file, 
        calibrated=not args.no_calibrated,
        save_plot=not args.no_plot
    )


if __name__ == '__main__':
    main()
