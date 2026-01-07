"""
Data Integration Script for EPL In-Game Prediction
Integrates pre-game data with event data to create comprehensive match datasets
"""
import os
import pandas as pd
import ast
import argparse
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class DataIntegrator:
    """Integrate pregame and event data into match statistics"""
    
    def __init__(self, project_dir=None):
        """
        Initialize DataIntegrator
        
        Args:
            project_dir: Project directory path (auto-detected if None)
        """
        if project_dir is None:
            self.PRJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.PRJ_DIR = project_dir
        
        self.pregame_data_path = os.path.join(self.PRJ_DIR, 'data_new', 'pregame_data', 'pregame_data.csv')
        self.event_data_dir = os.path.join(self.PRJ_DIR, 'data_new', 'event_data')
        self.match_output_dir = os.path.join(self.PRJ_DIR, 'data_new', 'match')
        self.full_output_path = os.path.join(self.PRJ_DIR, 'data_new', 'full.csv')
        
        # Create output directories
        os.makedirs(self.match_output_dir, exist_ok=True)
        
    def initialize_game_state(self, home_team_elo, away_team_elo):
        """
        Create initial game state at minute 0
        
        Args:
            home_team_elo: Home team ELO rating
            away_team_elo: Away team ELO rating
        
        Returns:
            Dictionary representing initial game state
        """
        return {
            'minute': 0,
            'half': 1,
            'ht_elo': home_team_elo,
            'at_elo': away_team_elo,
            'ht_goal': 0,
            'at_goal': 0,
            'pass': 0,
            'short_pass': 0,
            'long_pass': 0,
            'final_3rd_pass': 0,
            'key_pass': 0,
            'cross': 0,
            'corner': 0,
            'big_chance': 0,
            'shot': 0,
            'shot_6_yard_box': 0,
            'shot_penalty_box': 0,
            'shot_open_play': 0,
            'shot_fast_break': 0,
            'dispossessed': 0,
            'turnover': 0,
            'duel': 0,
            'tackle': 0,
            'interception': 0,
            'clearance': 0,
            'offside': 0,
            'yellow': 0,
            'red': 0,
            'result': 'D'
        }
    
    def process_event(self, event, new_event, ht_id):
        """
        Process a single event and update game state
        
        Args:
            event: Event data from event_data DataFrame
            new_event: Current game state dictionary
            ht_id: Home team ID
        """
        team_multiplier = 1 if event['teamId'] == ht_id else -1
        event_types = event['satisfiedEventsTypes']
        
        # Goals (type 16)
        if event['type']['value'] == 16:
            if 23 in event_types:  # Own goal
                if event['teamId'] == ht_id:
                    new_event['at_goal'] += 1
                else:
                    new_event['ht_goal'] += 1
            else:  # Regular goal
                if event['teamId'] == ht_id:
                    new_event['ht_goal'] += 1
                else:
                    new_event['at_goal'] += 1
        
        # Passes
        if 117 in event_types:
            new_event['pass'] += team_multiplier
        if 30 in event_types:
            new_event['short_pass'] += team_multiplier
        if 127 in event_types or 128 in event_types:
            new_event['long_pass'] += team_multiplier
        if 217 in event_types:
            new_event['final_3rd_pass'] += team_multiplier
        if 123 in event_types:
            new_event['key_pass'] += team_multiplier
        
        # Crosses and corners
        if 125 in event_types or 126 in event_types:
            new_event['cross'] += team_multiplier
        if 31 in event_types:
            new_event['corner'] += team_multiplier
        
        # Shots
        if 203 in event_types:
            new_event['big_chance'] += team_multiplier
        if 10 in event_types:
            new_event['shot'] += team_multiplier
        if 0 in event_types:
            new_event['shot_6_yard_box'] += team_multiplier
        if 1 in event_types:
            new_event['shot_penalty_box'] += team_multiplier
        if 3 in event_types:
            new_event['shot_open_play'] += team_multiplier
        if 4 in event_types:
            new_event['shot_fast_break'] += team_multiplier
        
        # Possession loss
        if 70 in event_types:
            new_event['dispossessed'] += team_multiplier
        if 69 in event_types:
            new_event['turnover'] += team_multiplier
        
        # Defensive actions
        if 197 in event_types:
            new_event['duel'] += team_multiplier
        if 143 in event_types:
            new_event['tackle'] += team_multiplier
        if 101 in event_types:
            new_event['interception'] += team_multiplier
        if 95 in event_types:
            new_event['clearance'] += team_multiplier
        
        # Discipline
        if 61 in event_types:
            new_event['offside'] += team_multiplier
        if 65 in event_types:
            new_event['yellow'] += team_multiplier
        if 68 in event_types:
            new_event['red'] += team_multiplier
    
    def integrate_match(self, match_row):
        """
        Integrate pregame and event data for a single match
        
        Args:
            match_row: Row from pregame_data DataFrame
        
        Returns:
            DataFrame with minute-by-minute match statistics
        """
        match_id = match_row['match_id']
        ht_id = match_row['home_team_id']
        at_id = match_row['away_team_id']
        ht_elo = match_row['home_team_elo']
        at_elo = match_row['away_team_elo']
        
        # Initialize game data with minute 0 state
        game_data = [self.initialize_game_state(ht_elo, at_elo)]
        
        # Load event data
        event_file = os.path.join(self.event_data_dir, f'{match_id}.csv')
        
        if not os.path.exists(event_file):
            logging.warning(f"Match {match_id}: Event data not found, skipping")
            return None
        
        try:
            event_data = pd.read_csv(event_file)
            
            # Parse JSON-like columns
            event_data['period'] = event_data['period'].apply(ast.literal_eval)
            event_data['type'] = event_data['type'].apply(ast.literal_eval)
            event_data['satisfiedEventsTypes'] = event_data['satisfiedEventsTypes'].apply(ast.literal_eval)
            
        except Exception as e:
            logging.error(f"Match {match_id}: Error loading event data - {e}")
            return None
        
        # Process events
        new_event = game_data[-1].copy()
        
        for idx, event in event_data.iterrows():
            new_event['minute'] = event['minute']
            new_event['half'] = event['period']['value']
            
            # Create new state when minute changes
            if new_event['minute'] > game_data[-1]['minute'] or \
               (new_event['half'] > game_data[-1]['half'] and new_event['half'] == 2):
                game_data.append(new_event.copy())
            
            # Process this event
            self.process_event(event, new_event, ht_id)
        
        # Create DataFrame
        df = pd.DataFrame(game_data)
        
        # Determine final result
        goal_diff = game_data[-1]['ht_goal'] - game_data[-1]['at_goal']
        if goal_diff > 0:
            df['result'] = 'W'
        elif goal_diff < 0:
            df['result'] = 'L'
        else:
            df['result'] = 'D'
        
        return df
    
    def integrate_all_matches(self, overwrite=False):
        """
        Integrate all matches from pregame data
        
        Args:
            overwrite: Whether to overwrite existing match files
        """
        logging.info("=" * 60)
        logging.info("Starting Data Integration")
        logging.info("=" * 60)
        
        # Load pregame data
        logging.info(f"Loading pregame data from {self.pregame_data_path}")
        pregame_data = pd.read_csv(self.pregame_data_path)
        logging.info(f"Found {len(pregame_data)} matches")
        
        # Remove existing full.csv if overwrite
        if overwrite and os.path.exists(self.full_output_path):
            os.remove(self.full_output_path)
            logging.info(f"Removed existing {self.full_output_path}")
        
        # Process each match
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for idx, match_row in tqdm(pregame_data.iterrows(), total=len(pregame_data), desc="Processing matches"):
            match_id = match_row['match_id']
            match_output_file = os.path.join(self.match_output_dir, f'{match_id}.csv')
            
            # Skip if file exists and not overwriting
            if not overwrite and os.path.exists(match_output_file):
                skipped_count += 1
                continue
            
            # Integrate match
            try:
                df = self.integrate_match(match_row)
                
                if df is None:
                    error_count += 1
                    continue
                
                # Save individual match file
                df.to_csv(match_output_file, index=False)
                
                # Append to full.csv
                if not os.path.isfile(self.full_output_path):
                    df.to_csv(self.full_output_path, mode='a', index=False, header=True)
                else:
                    df.to_csv(self.full_output_path, mode='a', index=False, header=False)
                
                processed_count += 1
                
            except Exception as e:
                logging.error(f"Match {match_id}: Unexpected error - {e}")
                error_count += 1
        
        # Summary
        logging.info("=" * 60)
        logging.info("Data Integration Complete")
        logging.info(f"Processed: {processed_count} matches")
        logging.info(f"Skipped: {skipped_count} matches (already exist)")
        logging.info(f"Errors: {error_count} matches")
        logging.info(f"Output: {self.full_output_path}")
        logging.info("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Integrate EPL match data')
    parser.add_argument('--project-dir', type=str, default=None,
                       help='Project directory (auto-detected if not specified)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing match files')
    
    args = parser.parse_args()
    
    # Create integrator and run
    integrator = DataIntegrator(project_dir=args.project_dir)
    integrator.integrate_all_matches(overwrite=args.overwrite)


if __name__ == '__main__':
    main()
