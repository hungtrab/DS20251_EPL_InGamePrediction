"""
Model definitions for EPL In-Game Prediction
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from config import RANDOM_SEED, NUM_CLASSES
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def get_base_classifiers():
    """Get list of base classifiers for initial evaluation"""
    classifiers = [
        LogisticRegression(multi_class='multinomial', solver='sag', max_iter=1000, random_state=RANDOM_SEED),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=RANDOM_SEED),
        GaussianNB(),
        RandomForestClassifier(random_state=RANDOM_SEED),
        AdaBoostClassifier(random_state=RANDOM_SEED),
        GradientBoostingClassifier(random_state=RANDOM_SEED),
        XGBClassifier(objective='multi:softprob', num_class=NUM_CLASSES, random_state=RANDOM_SEED),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()
    ]
    return classifiers


def get_classifier_by_name(name, **kwargs):
    """
    Get classifier instance by name
    
    Args:
        name: Name of the classifier
        **kwargs: Additional parameters for the classifier
    
    Returns:
        Classifier instance
    """
    classifiers_map = {
        'knn': KNeighborsClassifier,
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier,
        'adaboost': AdaBoostClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'naive_bayes': GaussianNB,
        'logistic_regression': LogisticRegression,
        'xgboost': XGBClassifier,
        'lda': LinearDiscriminantAnalysis,
        'qda': QuadraticDiscriminantAnalysis
    }
    
    if name not in classifiers_map:
        raise ValueError(f"Unknown classifier: {name}")
    
    # Set default random_state for classifiers that support it
    if name in ['decision_tree', 'random_forest', 'adaboost', 'gradient_boosting', 'xgboost']:
        kwargs.setdefault('random_state', RANDOM_SEED)
    
    # Special handling for XGBoost
    if name == 'xgboost':
        kwargs.setdefault('objective', 'multi:softprob')
        kwargs.setdefault('num_class', NUM_CLASSES)
    
    # Special handling for Logistic Regression
    if name == 'logistic_regression':
        kwargs.setdefault('multi_class', 'multinomial')
        kwargs.setdefault('max_iter', 1000)
    
    return classifiers_map[name](**kwargs)


def calibrate_classifier(classifier, X_calib, y_calib, method='isotonic', cv='prefit'):
    """
    Calibrate a classifier using CalibratedClassifierCV
    
    Args:
        classifier: Fitted classifier to calibrate
        X_calib: Calibration features
        y_calib: Calibration labels
        method: Calibration method ('isotonic' or 'sigmoid')
        cv: Cross-validation strategy
    
    Returns:
        Calibrated classifier
    """
    logging.info(f"Calibrating classifier using {method} method")
    
    calibrated = CalibratedClassifierCV(classifier, cv=cv, method=method)
    calibrated.fit(X_calib, y_calib)
    
    return calibrated
