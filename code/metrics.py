"""
Custom metrics for EPL In-Game Prediction
"""
import numpy as np
from sklearn.metrics import make_scorer, log_loss, accuracy_score
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def rps_score(outcomes, predictions):
    """
    Ranked Probability Score
    
    Args:
        outcomes: True labels (array of integers 0, 1, 2)
        predictions: Predicted probabilities (n_samples, 3)
    
    Returns:
        RPS score (higher is better, range 0-1)
    """
    loss = 0
    for i, p in enumerate(predictions):
        outcome = [1 if x == outcomes[i] else 0 for x in range(3)]
        tmp = probs = outs = 0
        for j, val in enumerate(predictions[i]):
            probs += val
            outs += outcome[j]
            tmp += (probs - outs) ** 2
        loss += tmp / 2
    loss /= len(predictions)
    return 1 - loss


def rps_loss(outcomes, predictions):
    """
    Ranked Probability Score Loss (lower is better)
    
    Args:
        outcomes: True labels
        predictions: Predicted probabilities
    
    Returns:
        RPS loss (lower is better)
    """
    return 1 - rps_score(outcomes, predictions)


def brier_score(outcomes, predictions):
    """
    Multi-class Brier Score
    
    Args:
        outcomes: True labels (array of integers)
        predictions: Predicted probabilities (n_samples, n_classes)
    
    Returns:
        Brier score (lower is better)
    """
    n_samples = len(outcomes)
    n_classes = predictions.shape[1]
    
    # One-hot encode outcomes
    y_true = np.zeros((n_samples, n_classes))
    y_true[np.arange(n_samples), outcomes] = 1
    
    # Compute Brier score
    score = np.mean(np.sum((predictions - y_true) ** 2, axis=1))
    
    return score


def compute_all_metrics(y_true, y_pred_proba, y_pred_class=None):
    """
    Compute all evaluation metrics
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred_class: Predicted classes (optional, will be computed if None)
    
    Returns:
        Dictionary of metrics
    """
    if y_pred_class is None:
        y_pred_class = np.argmax(y_pred_proba, axis=1)
    
    metrics = {
        'rps_score': rps_score(y_true, y_pred_proba),
        'rps_loss': rps_loss(y_true, y_pred_proba),
        'brier_score': brier_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred_class)
    }
    
    return metrics


# Create sklearn-compatible scorer
rps_scorer = make_scorer(rps_score, greater_is_better=True, needs_proba=True)
