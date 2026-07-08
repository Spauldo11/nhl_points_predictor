import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

# Add colors for easier terminal legibility
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

def time_to_seconds(time_str):
    """Converts a 'MM:SS' time string into a float of total seconds."""
    if pd.isna(time_str) or not isinstance(time_str, str):
        return 0.0
    if ':' in time_str:
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = int(parts[1])
        return float((minutes * 60) + seconds)
    try:
        return float(time_str)
    except ValueError:
        return 0.0

def predict_player_from_csv(player_name, season, csv_filepath="nhl_historical_stats.csv", 
                            model_path="nhl_point_predictor.keras", model_path2="nhl_point_predictorv2.keras", model_path3="nhl_points_predictorv3.keras", scaler_path="scaler.pkl"):
    """
    Loads the master CSV, replicates the training preprocessing pipeline,
    extracts the target player's row, and predicts next season's points.
    """
    # 1. Load the master dataset
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        return f"Error: Could not find {csv_filepath}"

    # 2. Replicate Training Preprocessing Steps
    # Convert time strings to numeric seconds
    time_cols = ['TOI', 'ATOI']
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(time_to_seconds)
            
    # Handle the position column (uncomment the line below if you used one-hot encoding)
    if 'Pos' in df.columns:
        df = pd.get_dummies(df, columns=['Pos'], dtype=int)

    # 3. Isolate the exact feature columns expected by the model
    exclude_cols = ['Player', 'Team', 'Season', 'Next_Season_PTS', 'Awards', 'xG', 'CF%', 'FF%', "+\-"] 
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Force all feature columns to be numeric types just like in training
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Filter the preprocessed dataframe for our target player and season
    player_data = df[(df['Player'] == player_name) & (df['Season'] == season)]
    
    if player_data.empty:
        return f"Error: Could not find stats for {player_name} in the {season} season."
    
    # Handle mid-season trade rows by prioritizing the 'TOT' (Total) row
    if len(player_data) > 1:
        if 'Team' in player_data.columns and 'TOT' in player_data['Team'].values:
            player_data = player_data[player_data['Team'] == 'TOT']
        else:
            player_data = player_data.iloc[[0]]

    # 5. Extract the clean numeric array
    player_row = player_data[feature_cols].copy()
    
    if player_row.isna().any().any():
        print(f"--> Notice: Found missing values for {player_name} in {season}. Imputing missing data...")
        
        # Loop through columns to apply intelligent default values
        for col in feature_cols:
            if player_row[col].isna().iloc[0]:
                # Strategy A: If it's an advanced or percentage metric, fill with column average
                if col in ['CF%', 'FF%', 'oiSH%', 'oZS%', 'xG']:
                    # Fill with the historical average of that stat across the entire dataset
                    league_average = df[col].mean()
                    player_row[col] = player_row[col].fillna(league_average)
                # Strategy B: If it's a counting or rate stat, default to 0
                else:
                    player_row[col] = player_row[col].fillna(0.0)

    # Convert the cleaned row into the final numpy array for the model
    stats_array = player_row.values
    
    # 6. Load the saved pipeline artifacts and predict
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        model = keras.models.load_model(model_path)
        model2 = keras.models.load_model(model_path2)
        model3 = keras.models.load_model(model_path3)
    except FileNotFoundError as e:
        return f"Error loading model or scaler files: {e}"

    # Scale and pass to the Keras model
    scaled_stats = scaler.transform(stats_array)
    prediction = model.predict(scaled_stats, verbose=0)
    alt_prediction = model2.predict(scaled_stats, verbose=0)
    alt_prediction2 = model3.predict(scaled_stats, verbose=0)

    
    # Return predicted value (clamped to 0 minimum)
    return (max(0.0, prediction[0][0]), alt_prediction[0][0], alt_prediction2[0][0], stats_array)

if __name__ == "__main__":
    target_player = input(f"Enter a player name you would like to predict the points for (ensure proper spelling and capitalization): {RED}")
    # Change base_season for testing
    # Use 2026 for actual next-year predictions
    base_season = 2025 

    prediction = predict_player_from_csv(target_player, base_season)
    
    print(f"{RESET}Running inference for {RED}{target_player}{RESET} using {YELLOW}{base_season}{RESET} metrics...")
    result1 = prediction[0]
    result2 = prediction[1]
    result3 = prediction[2]

    print(f"Stats used for prediction: {prediction[3]}")
    
    if isinstance(result1, str):
        print(result1)
    else:
        print("\n" + "="*50)
        print(f"Projected Points for {YELLOW}{base_season}-{base_season+1}{RESET} Season (100 Epochs): {result1:.1f}")
        print(f"Projected Points for {YELLOW}{base_season}-{base_season+1}{RESET} Season (500 Epochs): {result2:.1f}")
        print(f"Projected Points for {YELLOW}{base_season}-{base_season+1}{RESET} Season (500 Epochs | No +/-): {result3:.1f}")
        print("="*50)
