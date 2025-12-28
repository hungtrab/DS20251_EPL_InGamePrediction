"""
Data preprocessing utilities for EPL In-Game Prediction
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib
import os
from config import RESULTS_DIR
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class DataPreprocessor:
    """Advanced data preprocessing"""
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: Type of scaler ('standard', 'robust', 'minmax')
        """
        self.scaler_type = scaler_type
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.is_fitted = False
        
    def fit(self, X):
        """Fit the scaler on training data"""
        logging.info(f"Fitting {self.scaler_type} scaler")
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logging.info(f"Transforming data with {self.scaler_type} scaler")
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        """Fit and transform data"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform scaled data"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        return self.scaler.inverse_transform(X)
    
    def save(self, filename='preprocessor.pkl'):
        """Save fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        filepath = os.path.join(RESULTS_DIR, 'models', filename)
        joblib.dump(self, filepath)
        logging.info(f"Preprocessor saved to {filepath}")
        
    @staticmethod
    def load(filename='preprocessor.pkl'):
        """Load fitted preprocessor"""
        filepath = os.path.join(RESULTS_DIR, 'models', filename)
        logging.info(f"Loading preprocessor from {filepath}")
        return joblib.load(filepath)


class AdvancedFeatureProcessor:
    """Advanced feature engineering and processing"""
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.new_features = []
        
    def create_rolling_features(self, data, match_id_col=None, window_sizes=[3, 5, 10]):
        """
        Create rolling window features
        
        Args:
            data: DataFrame with features
            match_id_col: Column name for match ID grouping
            window_sizes: List of window sizes for rolling features
        """
        logging.info(f"Creating rolling features with windows: {window_sizes}")
        
        df = data.copy()
        
        # Features to compute rolling statistics on
        rolling_cols = ['shot', 'pass', 'corner', 'big_chance', 'key_pass', 'final_3rd_pass']
        
        for col in rolling_cols:
            if col not in df.columns:
                continue
                
            for window in window_sizes:
                if match_id_col:
                    # Group by match
                    df[f'{col}_rolling_{window}'] = df.groupby(match_id_col)[col].rolling(
                        window, min_periods=1
                    ).mean().reset_index(0, drop=True)
                    
                    df[f'{col}_rolling_std_{window}'] = df.groupby(match_id_col)[col].rolling(
                        window, min_periods=1
                    ).std().reset_index(0, drop=True).fillna(0)
                else:
                    df[f'{col}_rolling_{window}'] = df[col].rolling(
                        window, min_periods=1
                    ).mean()
                    
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(
                        window, min_periods=1
                    ).std().fillna(0)
                
                self.new_features.extend([f'{col}_rolling_{window}', f'{col}_rolling_std_{window}'])
        
        logging.info(f"Created {len(self.new_features)} rolling features")
        return df
    
    def create_interaction_features(self, data):
        """Create interaction features"""
        logging.info("Creating interaction features")
        
        df = data.copy()
        
        # ELO interactions
        if 'ht_elo' in df.columns and 'at_elo' in df.columns:
            df['elo_diff'] = df['ht_elo'] - df['at_elo']
            df['elo_sum'] = df['ht_elo'] + df['at_elo']
            df['elo_ratio'] = df['ht_elo'] / (df['at_elo'] + 1e-6)
            self.new_features.extend(['elo_diff', 'elo_sum', 'elo_ratio'])
        
        # Attack efficiency
        if 'shot' in df.columns and 'ht_goal' in df.columns and 'at_goal' in df.columns:
            df['goal_diff'] = df['ht_goal'] - df['at_goal']
            df['shot_efficiency'] = (df['ht_goal'] - df['at_goal']) / (abs(df['shot']) + 1)
            self.new_features.extend(['goal_diff', 'shot_efficiency'])
        
        # Possession proxy (from passes)
        if 'pass' in df.columns:
            df['possession_proxy'] = df['pass'] / (abs(df['pass']) + 1)
            self.new_features.append('possession_proxy')
        
        # Attacking pressure
        if 'shot' in df.columns and 'corner' in df.columns and 'big_chance' in df.columns:
            df['attacking_pressure'] = df['shot'] + df['corner'] + df['big_chance']
            self.new_features.append('attacking_pressure')
        
        # ELO-weighted shot
        if 'elo_diff' in df.columns and 'shot' in df.columns:
            df['elo_shot_interaction'] = df['elo_diff'] * df['shot']
            self.new_features.append('elo_shot_interaction')
        
        logging.info(f"Created interaction features")
        return df
    
    def create_temporal_features(self, data):
        """Create time-based features"""
        logging.info("Creating temporal features")
        
        df = data.copy()
        
        if 'minute' in df.columns:
            df['minutes_remaining'] = 90 - df['minute']
            df['is_early_game'] = (df['minute'] <= 15).astype(int)
            df['is_mid_game'] = ((df['minute'] > 15) & (df['minute'] <= 75)).astype(int)
            df['is_late_game'] = (df['minute'] > 75).astype(int)
            df['minute_squared'] = df['minute'] ** 2
            df['minute_log'] = np.log(df['minute'] + 1)
            
            self.new_features.extend([
                'minutes_remaining', 'is_early_game', 'is_mid_game', 
                'is_late_game', 'minute_squared', 'minute_log'
            ])
        
        if 'half' in df.columns:
            df['is_second_half'] = (df['half'] == 2).astype(int)
            self.new_features.append('is_second_half')
        
        logging.info(f"Created temporal features")
        return df
    
    def create_momentum_features(self, data, match_id_col=None):
        """Create momentum-based features"""
        logging.info("Creating momentum features")
        
        df = data.copy()
        
        # Goal momentum
        if 'ht_goal' in df.columns and 'at_goal' in df.columns:
            if match_id_col:
                df['goal_momentum'] = df.groupby(match_id_col)['ht_goal'].diff() - \
                                     df.groupby(match_id_col)['at_goal'].diff()
            else:
                df['goal_momentum'] = df['ht_goal'].diff() - df['at_goal'].diff()
            
            df['goal_momentum'] = df['goal_momentum'].fillna(0)
            self.new_features.append('goal_momentum')
        
        # Shot momentum
        if 'shot' in df.columns:
            if match_id_col:
                df['shot_momentum'] = df.groupby(match_id_col)['shot'].diff()
            else:
                df['shot_momentum'] = df['shot'].diff()
            
            df['shot_momentum'] = df['shot_momentum'].fillna(0)
            self.new_features.append('shot_momentum')
        
        logging.info(f"Created momentum features")
        return df
    
    def get_new_feature_names(self):
        """Get list of newly created features"""
        return self.new_features
