# Baseball Betting Model - Deployment Package

This repository contains all the files needed to deploy the Baseball Betting Model as a web application using Streamlit.

## Features

- **Pitcher Strikeout Prediction**: Predict strikeout probabilities for pitchers
- **Batter Hit Prediction**: Predict hit probabilities for batters
- **Matchup Analysis**: Analyze specific pitcher vs. batter matchups
- **Simulation**: Run simulations for multiple games to identify betting opportunities
- **Betting Edge Calculation**: Calculate betting edge and Kelly criterion stake sizing

## Deployment Instructions

### Option 1: Deploy to Streamlit Community Cloud (Recommended)

1. Create a GitHub account if you don't have one: https://github.com/join
2. Create a new repository on GitHub
3. Upload all files from this package to your repository
4. Sign up for Streamlit Community Cloud: https://streamlit.io/cloud
5. Connect your GitHub account to Streamlit Community Cloud
6. Deploy your app by selecting your repository, branch, and the main file (app.py)

### Option 2: Run Locally

1. Install Python 3.10 or newer
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Files Included

- `app.py`: Main application file
- `requirements.txt`: List of required Python packages
- `.github/workflows/streamlit-deploy.yml`: GitHub Actions workflow for continuous deployment

## Model Details

The baseball betting model uses machine learning algorithms to predict:
- Pitcher strikeout probabilities based on metrics like K/9, SwStr%, and WHIP
- Batter hit probabilities based on metrics like BA, Exit Velocity, and Launch Angle
- Ballpark effects to adjust predictions based on stadium characteristics

The betting algorithm calculates edge percentages and recommends optimal bet sizes using the Kelly Criterion.

## Support

For questions or issues, please open an issue on the GitHub repository.
