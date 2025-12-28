"""
Data loading utilities for EPL In-Game Prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, RESULT_MAP, RANDOM_SEED, TEST_SIZE
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class DataLoader:
    """Data loading and basic preparation"""
    
    def __init__(self, data_path=TRAIN_DATA_PATH):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_data(self):
        """Load data from CSV file"""
        logging.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        logging.info(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def prepare_features_labels(self):
        """Separate features and labels"""
        if self.data is None:
            self.load_data()
        
        self.X = self.data.drop(['result'], axis=1)
        self.y = self.data['result']
        self.feature_names = self.X.columns.tolist()
        
        # Convert to numpy arrays
        self.X = self.X.values
        
        # Encode labels
        self.y = np.array([RESULT_MAP[val] for val in self.y.values])
        
        logging.info(f"Features shape: {self.X.shape}")
        logging.info(f"Labels shape: {self.y.shape}")
        logging.info(f"Label distribution: {np.bincount(self.y)}")
        
        return self.X, self.y, self.feature_names
    
    def split_data(self, test_size=TEST_SIZE, random_state=RANDOM_SEED):
        """Split data into train and test sets"""
        if self.X is None or self.y is None:
            self.prepare_features_labels()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y
        )
        
        logging.info(f"Train set: {X_train.shape[0]} samples")
        logging.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_info(self):
        """Get data information"""
        if self.data is None:
            self.load_data()
        
        return {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'null_counts': self.data.isnull().sum().to_dict(),
            'description': self.data.describe().to_dict()
        }
