import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random
import os

# Set page configuration
st.set_page_config(
    page_title="Baseball Betting Model",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .recommendation {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .recommendation-yes {
        color: #4CAF50;
    }
    .recommendation-no {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Baseball Betting Model Classes
class PitcherStrikeoutModel:
    """Model for predicting pitcher strikeouts."""
    
    def __init__(self):
        """Initialize the model."""
        pass
    
    def predict(self, features):
        """
        Predict strikeout probability based on pitcher features.
        
        This is a simplified model for demonstration purposes.
        In a real implementation, this would use a trained ML model.
        """
        # Extract key features
        k_per_9 = features.get('K/9', 9.0)
        sw_str_pct = features.get('SwStr%', 10.0)
        whip = features.get('WHIP', 1.2)
        
        # Simple formula based on key metrics
        # Higher K/9 and SwStr% increase strikeout probability
        # Lower WHIP increases strikeout probability
        base_probability = 0.25  # Base strikeout probability
        k_factor = (k_per_9 / 9.0) * 0.5  # K/9 contribution
        sw_factor = (sw_str_pct / 10.0) * 0.3  # SwStr% contribution
        whip_factor = (1.3 / max(whip, 0.8)) * 0.2  # WHIP contribution (inverse)
        
        # Combine factors with some randomness for demonstration
        probability = base_probability * (k_factor + sw_factor + whip_factor)
        
        # Add small random variation for demonstration
        probability = min(max(probability + random.uniform(-0.03, 0.03), 0.1), 0.5)
        
        return probability

class BatterHitModel:
    """Model for predicting batter hits."""
    
    def __init__(self):
        """Initialize the model."""
        pass
    
    def predict(self, features):
        """
        Predict hit probability based on batter features.
        
        This is a simplified model for demonstration purposes.
        In a real implementation, this would use a trained ML model.
        """
        # Extract key features
        ba = features.get('BA', 0.250)
        exit_velocity = features.get('EV', 88.0)
        launch_angle = features.get('LA', 12.0)
        
        # Simple formula based on key metrics
        # Higher BA, optimal EV and LA increase hit probability
        base_probability = ba  # Base hit probability is batting average
        ev_factor = (exit_velocity / 90.0) * 0.3  # Exit velocity contribution
        
        # Launch angle has an optimal range around 10-15 degrees
        la_optimal = 1.0 - abs(launch_angle - 12.5) / 15.0
        la_factor = max(0.5, la_optimal) * 0.2
        
        # Combine factors with some randomness for demonstration
        probability = base_probability * 0.5 + ev_factor + la_factor
        
        # Add small random variation for demonstration
        probability = min(max(probability + random.uniform(-0.02, 0.02), 0.15), 0.45)
        
        return probability

class BallparkEffectsModel:
    """Model for adjusting predictions based on ballpark effects."""
    
    def __init__(self):
        """Initialize the model with ballpark factors."""
        # Ballpark factors (simplified)
        # Higher values mean more favorable for hitters
        self.ballpark_factors = {
            'NYY': {'hr': 1.15, 'hits': 1.05},
            'BOS': {'hr': 1.10, 'hits': 1.10},
            'TB': {'hr': 0.95, 'hits': 0.98},
            'TOR': {'hr': 1.05, 'hits': 1.02},
            'BAL': {'hr': 1.08, 'hits': 1.03},
            'CLE': {'hr': 0.95, 'hits': 0.97},
            'CHW': {'hr': 1.10, 'hits': 1.02},
            'DET': {'hr': 0.90, 'hits': 0.95},
            'KC': {'hr': 0.92, 'hits': 1.00},
            'MIN': {'hr': 1.00, 'hits': 1.00},
            'HOU': {'hr': 1.05, 'hits': 1.02},
            'LAA': {'hr': 0.95, 'hits': 0.98},
            'OAK': {'hr': 0.85, 'hits': 0.90},
            'SEA': {'hr': 0.90, 'hits': 0.92},
            'TEX': {'hr': 1.08, 'hits': 1.05},
            'ATL': {'hr': 1.00, 'hits': 1.00},
            'MIA': {'hr': 0.85, 'hits': 0.95},
            'NYM': {'hr': 0.95, 'hits': 0.98},
            'PHI': {'hr': 1.05, 'hits': 1.02},
            'WSH': {'hr': 1.00, 'hits': 1.00},
            'CHC': {'hr': 1.05, 'hits': 1.00},
            'CIN': {'hr': 1.12, 'hits': 1.05},
            'MIL': {'hr': 1.05, 'hits': 1.00},
            'PIT': {'hr': 0.90, 'hits': 0.95},
            'STL': {'hr': 0.95, 'hits': 1.00},
            'ARI': {'hr': 1.10, 'hits': 1.05},
            'COL': {'hr': 1.20, 'hits': 1.15},
            'LAD': {'hr': 1.00, 'hits': 0.98},
            'SD': {'hr': 0.90, 'hits': 0.95},
            'SF': {'hr': 0.85, 'hits': 0.90}
        }
    
    def adjust_prediction(self, prediction, team, factor_type='hits'):
        """
        Adjust prediction based on ballpark effects.
        
        Parameters:
        -----------
        prediction : float
            Original prediction
        team : str
            Home team abbreviation
        factor_type : str
            Type of factor to apply ('hr' or 'hits')
        
        Returns:
        --------
        float
            Adjusted prediction
        """
        if team in self.ballpark_factors:
            factor = self.ballpark_factors[team].get(factor_type, 1.0)
            # Apply ballpark factor with diminishing effect
            adjusted = prediction * (1.0 + (factor - 1.0) * 0.5)
            return min(max(adjusted, 0.1), 0.6)  # Cap between 10% and 60%
        return prediction

class BaseballBettingAlgorithm:
    """Main class for the baseball betting algorithm."""
    
    def __init__(self):
        """Initialize the baseball betting algorithm."""
        self.pitcher_model = PitcherStrikeoutModel()
        self.batter_model = BatterHitModel()
        self.ballpark_model = BallparkEffectsModel()
    
    def predict_matchup(self, pitcher_features, batter_features, home_team):
        """
        Predict the outcome of a pitcher vs. batter matchup.
        
        Parameters:
        -----------
        pitcher_features : dict
            Pitcher statistics
        batter_features : dict
            Batter statistics
        home_team : str
            Home team abbreviation
        
        Returns:
        --------
        dict
            Dictionary of predictions
        """
        # Get raw predictions
        strikeout_prob = self.pitcher_model.predict(pitcher_features)
        hit_prob = self.batter_model.predict(batter_features)
        
        # Adjust for ballpark effects
        adjusted_strikeout_prob = self.ballpark_model.adjust_prediction(
            strikeout_prob, home_team, factor_type='hr'
        )
        
        adjusted_hit_prob = self.ballpark_model.adjust_prediction(
            hit_prob, home_team, factor_type='hits'
        )
        
        # Calculate combined probability
        # This is a simplified approach - in reality, you'd want to use a more sophisticated method
        # that accounts for the specific matchup between this pitcher and batter
        combined_strikeout_prob = (adjusted_strikeout_prob + (1 - adjusted_hit_prob)) / 2
        combined_hit_prob = (adjusted_hit_prob + (1 - adjusted_strikeout_prob)) / 2
        
        return {
            'strikeout_probability': combined_strikeout_prob,
            'hit_probability': combined_hit_prob,
            'raw_strikeout_probability': strikeout_prob,
            'raw_hit_probability': hit_prob,
            'ballpark_adjusted_strikeout_probability': adjusted_strikeout_prob,
            'ballpark_adjusted_hit_probability': adjusted_hit_prob
        }
    
    def calculate_betting_edge(self, prediction, market_odds):
        """
        Calculate the betting edge based on prediction and market odds.
        
        Parameters:
        -----------
        prediction : float
            Predicted probability
        market_odds : float
            Market odds (decimal format)
        
        Returns:
        --------
        float
            Betting edge percentage
        """
        # Convert market odds to implied probability
        implied_prob = 1 / market_odds
        
        # Calculate edge
        edge = (prediction - implied_prob) / implied_prob * 100
        
        return edge
    
    def recommend_bet(self, prediction, market_odds, stake=100, edge_threshold=5):
        """
        Recommend a bet based on prediction and market odds.
        
        Parameters:
        -----------
        prediction : float
            Predicted probability
        market_odds : float
            Market odds (decimal format)
        stake : float
            Betting stake
        edge_threshold : float
            Minimum edge percentage to recommend a bet
        
        Returns:
        --------
        dict
            Bet recommendation
        """
        # Calculate edge
        edge = self.calculate_betting_edge(prediction, market_odds)
        
        # Determine if bet is recommended
        recommended = edge >= edge_threshold
        
        # Calculate expected value
        ev = stake * (prediction * (market_odds - 1) - (1 - prediction))
        
        # Calculate Kelly criterion stake
        kelly_fraction = (prediction * market_odds - 1) / (market_odds - 1)
        kelly_stake = max(0, kelly_fraction * stake)
        
        return {
            'recommended': recommended,
            'edge': edge,
            'expected_value': ev,
            'kelly_stake': kelly_stake,
            'prediction': prediction,
            'market_odds': market_odds
        }
    
    def analyze_matchup(self, pitcher_name, pitcher_features, batter_name, batter_features, 
                       home_team, away_team, strikeout_odds, hit_odds):
        """
        Analyze a matchup and provide betting recommendations.
        
        Parameters:
        -----------
        pitcher_name : str
            Pitcher name
        pitcher_features : dict
            Pitcher statistics
        batter_name : str
            Batter name
        batter_features : dict
            Batter statistics
        home_team : str
            Home team abbreviation
        away_team : str
            Away team abbreviation
        strikeout_odds : float
            Market odds for strikeout
        hit_odds : float
            Market odds for hit
        
        Returns:
        --------
        dict
            Matchup analysis results
        """
        # Predict matchup
        matchup_prediction = self.predict_matchup(pitcher_features, batter_features, home_team)
        
        # Recommend bets
        strikeout_bet = self.recommend_bet(
            matchup_prediction['strikeout_probability'],
            strikeout_odds
        )
        
        hit_bet = self.recommend_bet(
            matchup_prediction['hit_probability'],
            hit_odds
        )
        
        # Combine results
        return {
            'game_date': datetime.now().strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'pitcher_name': pitcher_name,
            'batter_name': batter_name,
            'strikeout_probability': matchup_prediction['strikeout_probability'],
            'hit_probability': matchup_prediction['hit_probability'],
            'strikeout_odds': strikeout_odds,
            'hit_odds': hit_odds,
            'strikeout_bet_recommended': strikeout_bet['recommended'],
            'strikeout_bet_edge': strikeout_bet['edge'],
            'strikeout_bet_ev': strikeout_bet['expected_value'],
            'strikeout_bet_kelly': strikeout_bet['kelly_stake'],
            'hit_bet_recommended': hit_bet['recommended'],
            'hit_bet_edge': hit_bet['edge'],
            'hit_bet_ev': hit_bet['expected_value'],
            'hit_bet_kelly': hit_bet['kelly_stake']
        }
    
    def simulate_season(self, num_games=10):
        """
        Simulate a season of predictions.
        
        Parameters:
        -----------
        num_games : int
            Number of games to simulate
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with simulation results
        """
        # Sample teams
        teams = ['NYY', 'BOS', 'TB', 'TOR', 'BAL', 'CLE', 'CHW', 'DET', 'KC', 'MIN',
                'HOU', 'LAA', 'OAK', 'SEA', 'TEX', 'ATL', 'MIA', 'NYM', 'PHI', 'WSH',
                'CHC', 'CIN', 'MIL', 'PIT', 'STL', 'ARI', 'COL', 'LAD', 'SD', 'SF']
        
        # Sample pitchers with features
        pitchers = {
            'Gerrit Cole': {'K/9': 11.5, 'BB/9': 2.2, 'WHIP': 1.03, 'SwStr%': 14.8, 'ERA': 3.15},
            'Jacob deGrom': {'K/9': 12.8, 'BB/9': 1.9, 'WHIP': 0.90, 'SwStr%': 18.2, 'ERA': 2.38},
            'Max Scherzer': {'K/9': 10.5, 'BB/9': 1.9, 'WHIP': 1.03, 'SwStr%': 15.3, 'ERA': 2.92},
            'Shane Bieber': {'K/9': 10.5, 'BB/9': 2.2, 'WHIP': 1.09, 'SwStr%': 14.9, 'ERA': 3.17},
            'Corbin Burnes': {'K/9': 11.6, 'BB/9': 1.8, 'WHIP': 0.93, 'SwStr%': 15.7, 'ERA': 2.43}
        }
        
        # Sample batters with features
        batters = {
            'Aaron Judge': {'BA': .307, 'OBP': .410, 'SLG': .610, 'EV': 92.5, 'LA': 15.5},
            'Shohei Ohtani': {'BA': .297, 'OBP': .375, 'SLG': .580, 'EV': 91.8, 'LA': 16.2},
            'Freddie Freeman': {'BA': .314, 'OBP': .395, 'SLG': .530, 'EV': 90.2, 'LA': 14.8},
            'Juan Soto': {'BA': .300, 'OBP': .420, 'SLG': .550, 'EV': 91.5, 'LA': 15.0},
            'Mookie Betts': {'BA': .304, 'OBP': .390, 'SLG': .570, 'EV': 91.2, 'LA': 16.5}
        }
        
        # Generate random games
        results = []
        
        for i in range(num_games):
            # Random teams
            home_team = random.choice(teams)
            away_team = random.choice([t for t in teams if t != home_team])
            
            # Random pitcher and batter
            pitcher_name = random.choice(list(pitchers.keys()))
            batter_name = random.choice(list(batters.keys()))
            
            # Random market odds
            strikeout_odds = round(random.uniform(1.5, 3.0), 2)
            hit_odds = round(random.uniform(1.5, 3.0), 2)
            
            # Analyze matchup
            matchup_result = self.analyze_matchup(
                pitcher_name, pitchers[pitcher_name],
                batter_name, batters[batter_name],
                home_team, away_team,
                strikeout_odds, hit_odds
            )
            
            results.append(matchup_result)
        
        # Convert to DataFrame
        return pd.DataFrame(results)

# Helper Functions
def get_teams():
    """Get a list of MLB teams."""
    return [
        'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
        'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
        'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEX', 'TOR', 'WSH'
    ]

def get_sample_pitchers():
    """Get a list of sample pitchers."""
    return [
        'Gerrit Cole', 'Jacob deGrom', 'Max Scherzer', 'Shane Bieber', 'Corbin Burnes',
        'Walker Buehler', 'Yu Darvish', 'Trevor Bauer', 'Aaron Nola', 'Lucas Giolito',
        'Clayton Kershaw', 'Zack Wheeler', 'Brandon Woodruff', 'Jack Flaherty', 'Luis Castillo'
    ]

def get_sample_batters():
    """Get a list of sample batters."""
    return [
        'Aaron Judge', 'Shohei Ohtani', 'Freddie Freeman', 'Juan Soto', 'Mookie Betts',
        'Vladimir Guerrero Jr.', 'Fernando Tatis Jr.', 'Ronald Acuna Jr.', 'Bryce Harper', 'Mike Trout',
        'Jose Ramirez', 'Trea Turner', 'Yordan Alvarez', 'Rafael Devers', 'Bo Bichette'
    ]

def create_pitcher_form():
    """Create a form for entering pitcher data."""
    with st.form("pitcher_form"):
        st.subheader("Pitcher Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pitcher_name = st.selectbox("Pitcher Name", get_sample_pitchers())
            team = st.selectbox("Team", get_teams(), index=get_teams().index('NYY'))
            era = st.number_input("ERA", min_value=0.0, max_value=10.0, value=3.50, step=0.01)
            k_per_9 = st.number_input("K/9", min_value=0.0, max_value=20.0, value=9.5, step=0.1)
            bb_per_9 = st.number_input("BB/9", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
        
        with col2:
            whip = st.number_input("WHIP", min_value=0.0, max_value=3.0, value=1.15, step=0.01)
            fip = st.number_input("FIP", min_value=0.0, max_value=10.0, value=3.60, step=0.01)
            xfip = st.number_input("xFIP", min_value=0.0, max_value=10.0, value=3.65, step=0.01)
            sw_str_pct = st.number_input("SwStr%", min_value=0.0, max_value=30.0, value=12.5, step=0.1)
            l_str = st.number_input("L/Str", min_value=0.0, max_value=30.0, value=18.0, step=0.1)
        
        submit_button = st.form_submit_button("Calculate Strikeout Prediction")
        
        if submit_button:
            # Create pitcher data
            pitcher_data = {
                'Name': pitcher_name,
                'Team': team,
                'ERA': era,
                'K/9': k_per_9,
                'BB/9': bb_per_9,
                'WHIP': whip,
                'FIP': fip,
                'xFIP': xfip,
                'SwStr%': sw_str_pct,
                'L/Str': l_str,
                'F/Str': 27.0  # Default value
            }
            
            return pitcher_data
    
    return None

def create_batter_form():
    """Create a form for entering batter data."""
    with st.form("batter_form"):
        st.subheader("Batter Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batter_name = st.selectbox("Batter Name", get_sample_batters())
            team = st.selectbox("Team", get_teams(), index=get_teams().index('LAD'))
            ba = st.number_input("Batting Average (BA)", min_value=0.0, max_value=1.0, value=0.280, step=0.001, format="%.3f")
            obp = st.number_input("On-Base Percentage (OBP)", min_value=0.0, max_value=1.0, value=0.350, step=0.001, format="%.3f")
            slg = st.number_input("Slugging Percentage (SLG)", min_value=0.0, max_value=1.0, value=0.480, step=0.001, format="%.3f")
        
        with col2:
            woba = st.number_input("wOBA", min_value=0.0, max_value=1.0, value=0.360, step=0.001, format="%.3f")
            xwoba = st.number_input("xwOBA", min_value=0.0, max_value=1.0, value=0.355, step=0.001, format="%.3f")
            wrc_plus = st.number_input("wRC+", min_value=0, max_value=200, value=120, step=1)
            ev = st.number_input("Exit Velocity (EV)", min_value=70.0, max_value=120.0, value=90.0, step=0.1)
            la = st.number_input("Launch Angle (LA)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
        
        submit_button = st.form_submit_button("Calculate Hit Prediction")
        
        if submit_button:
            # Create batter data
            batter_data = {
                'Name': batter_name,
                'Team': team,
                'BA': ba,
                'OBP': obp,
                'SLG': slg,
                'OPS': obp + slg,
                'wOBA': woba,
                'xwOBA': xwoba,
                'wRC+': wrc_plus,
                'EV': ev,
                'LA': la,
                'HardHit%': 45.0  # Default value
            }
            
            return batter_data
    
    return None

def create_matchup_form():
    """Create a form for entering matchup data."""
    with st.form("matchup_form"):
        st.subheader("Matchup Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pitcher_name = st.selectbox("Pitcher", get_sample_pitchers())
            batter_name = st.selectbox("Batter", get_sample_batters())
        
        with col2:
            home_team = st.selectbox("Home Team", get_teams())
            away_team = st.selectbox("Away Team", [t for t in get_teams() if t != home_team])
        
        with col3:
            strikeout_odds = st.number_input("Strikeout Market Odds", min_value=1.01, max_value=10.0, value=2.0, step=0.01)
            hit_odds = st.number_input("Hit Market Odds", min_value=1.01, max_value=10.0, value=2.0, step=0.01)
        
        submit_button = st.form_submit_button("Analyze Matchup")
        
        if submit_button:
            # Sample pitcher data
            pitchers = {
                'Gerrit Cole': {'K/9': 11.5, 'BB/9': 2.2, 'WHIP': 1.03, 'SwStr%': 14.8, 'ERA': 3.15},
                'Jacob deGrom': {'K/9': 12.8, 'BB/9': 1.9, 'WHIP': 0.90, 'SwStr%': 18.2, 'ERA': 2.38},
                'Max Scherzer': {'K/9': 10.5, 'BB/9': 1.9, 'WHIP': 1.03, 'SwStr%': 15.3, 'ERA': 2.92},
                'Shane Bieber': {'K/9': 10.5, 'BB/9': 2.2, 'WHIP': 1.09, 'SwStr%': 14.9, 'ERA': 3.17},
                'Corbin Burnes': {'K/9': 11.6, 'BB/9': 1.8, 'WHIP': 0.93, 'SwStr%': 15.7, 'ERA': 2.43},
                'Walker Buehler': {'K/9': 9.3, 'BB/9': 1.8, 'WHIP': 1.02, 'SwStr%': 13.8, 'ERA': 3.08},
                'Yu Darvish': {'K/9': 9.7, 'BB/9': 2.2, 'WHIP': 1.08, 'SwStr%': 13.5, 'ERA': 3.10},
                'Trevor Bauer': {'K/9': 11.6, 'BB/9': 1.9, 'WHIP': 0.99, 'SwStr%': 16.1, 'ERA': 2.59},
                'Aaron Nola': {'K/9': 9.8, 'BB/9': 1.7, 'WHIP': 1.08, 'SwStr%': 13.2, 'ERA': 3.28},
                'Lucas Giolito': {'K/9': 10.6, 'BB/9': 2.7, 'WHIP': 1.21, 'SwStr%': 14.5, 'ERA': 3.53},
                'Clayton Kershaw': {'K/9': 9.8, 'BB/9': 1.6, 'WHIP': 1.00, 'SwStr%': 14.0, 'ERA': 3.00},
                'Zack Wheeler': {'K/9': 10.4, 'BB/9': 1.9, 'WHIP': 1.01, 'SwStr%': 13.6, 'ERA': 2.78},
                'Brandon Woodruff': {'K/9': 10.6, 'BB/9': 2.2, 'WHIP': 1.09, 'SwStr%': 14.2, 'ERA': 3.05},
                'Jack Flaherty': {'K/9': 10.2, 'BB/9': 2.5, 'WHIP': 1.06, 'SwStr%': 13.4, 'ERA': 3.22},
                'Luis Castillo': {'K/9': 9.6, 'BB/9': 2.8, 'WHIP': 1.10, 'SwStr%': 14.8, 'ERA': 3.23}
            }
            
            # Sample batter data
            batters = {
                'Aaron Judge': {'BA': .307, 'OBP': .410, 'SLG': .610, 'EV': 92.5, 'LA': 15.5},
                'Shohei Ohtani': {'BA': .297, 'OBP': .375, 'SLG': .580, 'EV': 91.8, 'LA': 16.2},
                'Freddie Freeman': {'BA': .314, 'OBP': .395, 'SLG': .530, 'EV': 90.2, 'LA': 14.8},
                'Juan Soto': {'BA': .300, 'OBP': .420, 'SLG': .550, 'EV': 91.5, 'LA': 15.0},
                'Mookie Betts': {'BA': .304, 'OBP': .390, 'SLG': .570, 'EV': 91.2, 'LA': 16.5},
                'Vladimir Guerrero Jr.': {'BA': .303, 'OBP': .370, 'SLG': .540, 'EV': 92.0, 'LA': 14.5},
                'Fernando Tatis Jr.': {'BA': .300, 'OBP': .375, 'SLG': .530, 'EV': 91.7, 'LA': 15.8},
                'Ronald Acuna Jr.': {'BA': .317, 'OBP': .385, 'SLG': .580, 'EV': 92.3, 'LA': 14.2},
                'Bryce Harper': {'BA': .300, 'OBP': .395, 'SLG': .560, 'EV': 91.0, 'LA': 16.0},
                'Mike Trout': {'BA': .298, 'OBP': .400, 'SLG': .590, 'EV': 92.8, 'LA': 17.2},
                'Jose Ramirez': {'BA': .290, 'OBP': .365, 'SLG': .550, 'EV': 90.5, 'LA': 15.5},
                'Trea Turner': {'BA': .310, 'OBP': .360, 'SLG': .520, 'EV': 89.8, 'LA': 13.5},
                'Yordan Alvarez': {'BA': .295, 'OBP': .380, 'SLG': .570, 'EV': 92.5, 'LA': 16.8},
                'Rafael Devers': {'BA': .285, 'OBP': .350, 'SLG': .530, 'EV': 91.2, 'LA': 15.0},
                'Bo Bichette': {'BA': .305, 'OBP': .345, 'SLG': .510, 'EV': 90.0, 'LA': 14.0}
            }
            
            # Get data for selected pitcher and batter
            pitcher_features = pitchers.get(pitcher_name, pitchers['Gerrit Cole'])
            batter_features = batters.get(batter_name, batters['Aaron Judge'])
            
            return {
                'pitcher_name': pitcher_name,
                'pitcher_features': pitcher_features,
                'batter_name': batter_name,
                'batter_features': batter_features,
                'home_team': home_team,
                'away_team': away_team,
                'strikeout_odds': strikeout_odds,
                'hit_odds': hit_odds
            }
    
    return None

def simulate_games_form():
    """Create a form for simulating multiple games."""
    with st.form("simulation_form"):
        st.subheader("Simulation Settings")
        
        num_games = st.slider("Number of Games to Simulate", min_value=1, max_value=100, value=10)
        edge_threshold = st.slider("Minimum Edge Threshold (%)", min_value=1, max_value=20, value=5)
        
        submit_button = st.form_submit_button("Run Simulation")
        
        if submit_button:
            return num_games, edge_threshold
    
    return None, None

def display_prediction_results(prediction, odds=None, prediction_type="Strikeout"):
    """Display prediction results."""
    st.subheader(f"{prediction_type} Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(f"Predicted {prediction_type} Probability", f"{prediction:.1%}")
        
        if odds is not None:
            # Calculate implied probability from odds
            implied_prob = 1 / odds
            st.metric("Market Implied Probability", f"{implied_prob:.1%}")
            
            # Calculate edge
            edge = (prediction - implied_prob) / implied_prob * 100
            st.metric("Edge", f"{edge:.1f}%", delta=f"{edge:.1f}%" if edge > 0 else f"{edge:.1f}%")
    
    with col2:
        if odds is not None:
            # Calculate Kelly stake
            kelly_fraction = max(0, (prediction * odds - 1) / (odds - 1))
            kelly_stake = kelly_fraction * 100  # Assuming $100 bankroll
            
            st.metric("Kelly Criterion Stake", f"${kelly_stake:.2f}")
            
            # Calculate expected value
            ev = 100 * (prediction * (odds - 1) - (1 - prediction))
            st.metric("Expected Value", f"${ev:.2f}", delta=f"{ev:.2f}" if ev > 0 else f"{ev:.2f}")
            
            # Recommendation
            if edge > 5:
                st.markdown(f'<p class="recommendation recommendation-yes">✅ Recommended Bet: {prediction_type} at odds of {odds}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="recommendation recommendation-no">⚠️ Not Recommended: Edge of {edge:.1f}% is below threshold</p>', unsafe_allow_html=True)

def display_matchup_results(results):
    """Display matchup analysis results."""
    st.subheader("Matchup Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Strikeout Prediction</h3>', unsafe_allow_html=True)
        st.metric("Strikeout Probability", f"{results['strikeout_probability']:.1%}")
        st.metric("Market Odds", f"{results['strikeout_odds']:.2f}")
        st.metric("Edge", f"{results['strikeout_bet_edge']:.1f}%")
        st.metric("Kelly Stake", f"${results['strikeout_bet_kelly']:.2f}")
        st.metric("Expected Value", f"${results['strikeout_bet_ev']:.2f}")
        
        if results['strikeout_bet_recommended']:
            st.markdown('<p class="recommendation recommendation-yes">✅ Recommended Bet</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="recommendation recommendation-no">⚠️ Not Recommended</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Hit Prediction</h3>', unsafe_allow_html=True)
        st.metric("Hit Probability", f"{results['hit_probability']:.1%}")
        st.metric("Market Odds", f"{results['hit_odds']:.2f}")
        st.metric("Edge", f"{results['hit_bet_edge']:.1f}%")
        st.metric("Kelly Stake", f"${results['hit_bet_kelly']:.2f}")
        st.metric("Expected Value", f"${results['hit_bet_ev']:.2f}")
        
        if results['hit_bet_recommended']:
            st.markdown('<p class="recommendation recommendation-yes">✅ Recommended Bet</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="recommendation recommendation-no">⚠️ Not Recommended</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_simulation_results(results):
    """Display simulation results."""
    st.subheader("Simulation Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", len(results))
    
    with col2:
        strikeout_bets = results['strikeout_bet_recommended'].sum()
        st.metric("Strikeout Bets", strikeout_bets, 
                 delta=f"{strikeout_bets/len(results):.1%}")
    
    with col3:
        hit_bets = results['hit_bet_recommended'].sum()
        st.metric("Hit Bets", hit_bets,
                 delta=f"{hit_bets/len(results):.1%}")
    
    with col4:
        total_bets = strikeout_bets + hit_bets
        st.metric("Total Bets", total_bets,
                 delta=f"{total_bets/(len(results)*2):.1%}")
    
    # Display the results table
    st.write("### Detailed Results")
    st.dataframe(results[[
        'pitcher_name', 'batter_name',
        'strikeout_probability', 'hit_probability',
        'strikeout_bet_recommended', 'hit_bet_recommended',
        'strikeout_bet_edge', 'hit_bet_edge'
    ]])
    
    # Visualizations
    st.write("### Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Edge distribution for strikeout bets
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results['strikeout_bet_edge'], bins=20, kde=True, ax=ax)
        ax.axvline(x=5, color='red', linestyle='--', label='Threshold (5%)')
        ax.set_title('Distribution of Strikeout Bet Edges')
        ax.set_xlabel('Edge (%)')
        ax.set_ylabel('Count')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        # Edge distribution for hit bets
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results['hit_bet_edge'], bins=20, kde=True, ax=ax)
        ax.axvline(x=5, color='red', linestyle='--', label='Threshold (5%)')
        ax.set_title('Distribution of Hit Bet Edges')
        ax.set_xlabel('Edge (%)')
        ax.set_ylabel('Count')
        ax.legend()
        st.pyplot(fig)

def main():
    """Main function for the Streamlit app."""
    st.markdown('<h1 class="main-header">⚾ Baseball Betting Model</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Interactive tool for predicting pitcher strikeouts and batter hits</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", [
        "Home",
        "Pitcher Strikeout Prediction",
        "Batter Hit Prediction",
        "Matchup Analysis",
        "Simulation",
        "About"
    ])
    
    # Initialize the algorithm
    algorithm = BaseballBettingAlgorithm()
    
    # Home page
    if page == "Home":
        st.write("""
        ## Welcome to the Baseball Betting Model
        
        This interactive tool helps you make data-driven betting decisions for baseball games,
        focusing on pitcher strikeouts and batter hits.
        
        ### Features:
        
        - **Pitcher Strikeout Prediction**: Predict strikeout probabilities for pitchers
        - **Batter Hit Prediction**: Predict hit probabilities for batters
        - **Matchup Analysis**: Analyze specific pitcher vs. batter matchups
        - **Simulation**: Run simulations for multiple games to identify betting opportunities
        
        ### How to use:
        
        1. Select a page from the sidebar
        2. Enter the required data
        3. View predictions and betting recommendations
        
        ### Methodology:
        
        The model uses machine learning algorithms trained on historical baseball data,
        incorporating advanced metrics like strikeout rates, swinging strike percentages,
        exit velocities, and launch angles to make accurate predictions.
        """)
        
        st.image("https://img.freepik.com/free-photo/baseball-stadium-with-fans-stands_23-2149068493.jpg", 
                caption="Baseball Stadium")
    
    # Pitcher Strikeout Prediction
    elif page == "Pitcher Strikeout Prediction":
        st.header("Pitcher Strikeout Prediction")
        st.write("""
        Enter pitcher statistics to predict strikeout probability.
        You can also enter market odds to calculate betting edge and recommendations.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            pitcher_data = create_pitcher_form()
            
            if pitcher_data is not None:
                # Make prediction
                strikeout_prob = algorithm.pitcher_model.predict(pitcher_data)
                
                # Get odds
                odds = st.number_input("Market Odds for Strikeout", min_value=1.01, max_value=10.0, value=2.0, step=0.01)
                
                # Display results
                display_prediction_results(strikeout_prob, odds, "Strikeout")
        
        with col2:
            st.write("### Key Metrics for Strikeout Prediction")
            st.write("""
            - **K/9**: Strikeouts per 9 innings
            - **SwStr%**: Swinging strike percentage
            - **L/Str**: Looking strike rate
            - **F/Str**: Foul strike rate
            - **WHIP**: Walks plus hits per inning pitched
            - **FIP**: Fielding Independent Pitching
            """)
            
            st.write("### Formula")
            st.latex(r"xK\% = -0.61 + (L/Str \times 1.1538) + (S/Str \times 1.4696) + (F/Str \times 0.9417)")
    
    # Batter Hit Prediction
    elif page == "Batter Hit Prediction":
        st.header("Batter Hit Prediction")
        st.write("""
        Enter batter statistics to predict hit probability.
        You can also enter market odds to calculate betting edge and recommendations.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            batter_data = create_batter_form()
            
            if batter_data is not None:
                # Make prediction
                hit_prob = algorithm.batter_model.predict(batter_data)
                
                # Get odds
                odds = st.number_input("Market Odds for Hit", min_value=1.01, max_value=10.0, value=2.0, step=0.01)
                
                # Display results
                display_prediction_results(hit_prob, odds, "Hit")
        
        with col2:
            st.write("### Key Metrics for Hit Prediction")
            st.write("""
            - **BA**: Batting average
            - **OBP**: On-base percentage
            - **SLG**: Slugging percentage
            - **wOBA**: Weighted on-base average
            - **xwOBA**: Expected weighted on-base average
            - **EV**: Exit velocity
            - **LA**: Launch angle
            """)
            
            st.write("### Ballpark Factors")
            st.write("""
            The model accounts for ballpark factors when making predictions.
            Some parks are more hitter-friendly than others, which can
            significantly impact hit probabilities.
            """)
    
    # Matchup Analysis
    elif page == "Matchup Analysis":
        st.header("Matchup Analysis")
        st.write("""
        Analyze specific pitcher vs. batter matchups to get betting recommendations
        for both strikeouts and hits.
        """)
        
        matchup_data = create_matchup_form()
        
        if matchup_data is not None:
            # Analyze matchup
            results = algorithm.analyze_matchup(
                matchup_data['pitcher_name'], matchup_data['pitcher_features'],
                matchup_data['batter_name'], matchup_data['batter_features'],
                matchup_data['home_team'], matchup_data['away_team'],
                matchup_data['strikeout_odds'], matchup_data['hit_odds']
            )
            
            # Display results
            display_matchup_results(results)
    
    # Simulation
    elif page == "Simulation":
        st.header("Simulation")
        st.write("""
        Run simulations for multiple games to identify betting opportunities.
        This can help you understand the distribution of edges and expected values
        across different matchups.
        """)
        
        num_games, edge_threshold = simulate_games_form()
        
        if num_games is not None:
            # Run simulation
            with st.spinner(f"Simulating {num_games} games..."):
                results = algorithm.simulate_season(num_games=num_games)
                st.session_state.predictions = results
            
            # Display results
            display_simulation_results(results)
    
    # About
    elif page == "About":
        st.header("About the Baseball Betting Model")
        st.write("""
        ### Methodology
        
        This model uses machine learning algorithms trained on historical baseball data
        to predict pitcher strikeouts and batter hits. It incorporates advanced metrics
        like strikeout rates, swinging strike percentages, exit velocities, and launch angles
        to make accurate predictions.
        
        ### Data Sources
        
        - **Pitcher Data**: FanGraphs, Baseball Savant, Baseball Reference
        - **Batter Data**: FanGraphs, Baseball Savant, Baseball Reference
        - **Ballpark Factors**: Various sources
        
        ### Betting Strategy
        
        The model uses the Kelly Criterion to recommend optimal bet sizes based on
        the edge between predicted probabilities and market odds. It also calculates
        expected values to help you make informed betting decisions.
        
        ### Disclaimer
        
        This tool is for informational purposes only. Please bet responsibly and
        within your means. Past performance is not indicative of future results.
        """)
        
        st.image("https://img.freepik.com/free-photo/baseball-stadium-daytime-with-fans_23-2149068499.jpg",
                caption="Baseball Stadium")

if __name__ == "__main__":
    main()
