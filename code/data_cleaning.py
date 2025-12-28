"""
Data Cleaning Script for EPL In-Game Prediction
Cleans integrated match data and splits into train/test sets
"""
import os
import random
import pandas as pd
import argparse
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class DataCleaner:
    """Clean and prepare match data for modeling"""
    
    def __init__(self, project_dir=None, random_seed=42):
        """
        Initialize DataCleaner
        
        Args:
            project_dir: Project directory path (auto-detected if None)
            random_seed: Random seed for reproducibility
        """
        if project_dir is None:
            self.PRJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.PRJ_DIR = project_dir
        
        self.random_seed = random_seed
        random.seed(random_seed)
        
        self.match_data_dir = os.path.join(self.PRJ_DIR, 'data', 'match')
        self.train_dir = os.path.join(self.PRJ_DIR, 'data', 'train')
        self.test_dir = os.path.join(self.PRJ_DIR, 'data', 'test')
        
        # Create output directories
        os.makedirs(os.path.join(self.train_dir, 'match'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'match'), exist_ok=True)
        
        # Features to drop (low impact based on EDA)
        self.drop_features = [
            'short_pass', 'long_pass', 'cross', 
            'shot_6_yard_box', 'shot_penalty_box',
            'shot_open_play', 'shot_fast_break', 
            'dispossessed', 'turnover', 'duel',
            'tackle', 'interception', 'clearance', 
            'offside', 'yellow'
        ]
    
    def clean_match(self, match_data):
        """
        Clean a single match data
        
        Args:
            match_data: DataFrame with raw match data
        
        Returns:
            Cleaned DataFrame
        """
        # Drop low-impact features
        match_data = match_data.drop(columns=self.drop_features, errors='ignore')
        
        new_match_data = pd.DataFrame()
        min_cnt = 0
        half_cnt = 1
        
        for idx, state in match_data.iterrows():
            minute = state['minute']
            half = state['half']
            
            # Eliminate error data (extra time beyond normal limits)
            if half > 2 or minute > 107:
                continue
            
            # Fill missing minutes with previous state
            while min_cnt != minute:
                if half_cnt != half:
                    half_cnt += 1
                    min_cnt = 45
                    continue
                
                # Use previous state for missing minute
                if len(new_match_data) > 0:
                    new_state = new_match_data.iloc[-1].copy()
                    new_state['minute'] = min_cnt
                    new_match_data = pd.concat(
                        [new_match_data, pd.DataFrame([new_state])], 
                        ignore_index=True
                    )
                
                min_cnt += 1
            
            # Add current state
            new_match_data = pd.concat(
                [new_match_data, pd.DataFrame([state])], 
                ignore_index=True
            )
            min_cnt += 1
        
        return new_match_data
    
    def split_data(self, test_ratio=0.2):
        """
        Split match data into train and test sets
        
        Args:
            test_ratio: Ratio of test set (default 0.2 = 20%)
        
        Returns:
            Tuple of (train_list, test_list)
        """
        logging.info(f"Splitting data with test_ratio={test_ratio}")
        
        # Get all match files
        match_list = os.listdir(self.match_data_dir)
        match_list = [f for f in match_list if f.endswith('.csv')]
        
        logging.info(f"Found {len(match_list)} match files")
        
        # Shuffle
        random.shuffle(match_list)
        
        # Split
        test_size = int(len(match_list) * test_ratio)
        train_list = match_list[:-test_size]
        test_list = match_list[-test_size:]
        
        logging.info(f"Train set: {len(train_list)} matches")
        logging.info(f"Test set: {len(test_list)} matches")
        
        return train_list, test_list
    
    def process_dataset(self, match_list, output_dir, dataset_name):
        """
        Process a list of matches and save to output directory
        
        Args:
            match_list: List of match filenames
            output_dir: Output directory path
            dataset_name: Name of dataset (for logging)
        
        Returns:
            Full DataFrame with all processed matches
        """
        logging.info(f"Processing {dataset_name} dataset...")
        
        full_data = pd.DataFrame()
        error_count = 0
        
        for match_file in tqdm(match_list, desc=f"Processing {dataset_name}"):
            try:
                # Load match data
                match_path = os.path.join(self.match_data_dir, match_file)
                match_data = pd.read_csv(match_path)
                
                # Clean match
                cleaned_data = self.clean_match(match_data)
                
                # Save individual match
                output_match_path = os.path.join(output_dir, 'match', match_file)
                cleaned_data.to_csv(output_match_path, index=False)
                
                # Append to full dataset
                full_data = pd.concat([full_data, cleaned_data], ignore_index=True)
                
            except Exception as e:
                logging.error(f"Error processing {match_file}: {e}")
                error_count += 1
        
        # Save full dataset
        full_output_path = os.path.join(output_dir, 'data.csv')
        full_data.to_csv(full_output_path, index=False)
        
        logging.info(f"{dataset_name} dataset complete:")
        logging.info(f"  Matches: {len(match_list) - error_count}/{len(match_list)}")
        logging.info(f"  Total samples: {len(full_data)}")
        logging.info(f"  Output: {full_output_path}")
        
        if error_count > 0:
            logging.warning(f"  Errors: {error_count} matches")
        
        return full_data
    
    def clean_and_split(self, test_ratio=0.2):
        """
        Main method to clean all data and split into train/test
        
        Args:
            test_ratio: Ratio of test set
        """
        logging.info("=" * 60)
        logging.info("Starting Data Cleaning and Splitting")
        logging.info("=" * 60)
        
        # Split data
        train_list, test_list = self.split_data(test_ratio=test_ratio)
        
        # Process training data
        train_data = self.process_dataset(train_list, self.train_dir, "Training")
        
        # Process test data
        test_data = self.process_dataset(test_list, self.test_dir, "Test")
        
        # Summary
        logging.info("=" * 60)
        logging.info("Data Cleaning Complete")
        logging.info(f"Training samples: {len(train_data)}")
        logging.info(f"Test samples: {len(test_data)}")
        logging.info(f"Features: {len(train_data.columns) - 1}")  # -1 for 'result' column
        logging.info(f"Dropped features: {', '.join(self.drop_features)}")
        logging.info("=" * 60)
        
        # Show feature summary
        logging.info("\nFinal features:")
        for col in train_data.columns:
            logging.info(f"  - {col}")
        
        # Show class distribution
        logging.info("\nClass distribution:")
        for dataset_name, data in [("Train", train_data), ("Test", test_data)]:
            if 'result' in data.columns:
                dist = data['result'].value_counts()
                logging.info(f"  {dataset_name}: W={dist.get('W', 0)}, "
                           f"D={dist.get('D', 0)}, L={dist.get('L', 0)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Clean EPL match data')
    parser.add_argument('--project-dir', type=str, default=None,
                       help='Project directory (auto-detected if not specified)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Test set ratio (default: 0.2)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create cleaner and run
    cleaner = DataCleaner(
        project_dir=args.project_dir,
        random_seed=args.random_seed
    )
    cleaner.clean_and_split(test_ratio=args.test_ratio)


if __name__ == '__main__':
    main()
