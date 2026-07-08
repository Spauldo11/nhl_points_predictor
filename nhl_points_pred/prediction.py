import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

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
                            model_path="nhl_point_predictor.keras", scaler_path="scaler.pkl"):
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
    # (Ensure this list matches your training script perfectly!)
    exclude_cols = ['Player', 'Team', 'Season', 'Next_Season_PTS', 'Awards', 'xG', 'CF%', 'FF%'] 
    # If you chose to drop position instead of encoding it, uncomment the line below:
    # exclude_cols.append('Pos') 
    
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
    except FileNotFoundError as e:
        return f"Error loading model or scaler files: {e}"

    # Scale and pass to the Keras model
    scaled_stats = scaler.transform(stats_array)
    prediction = model.predict(scaled_stats, verbose=0)
    
    # Return predicted value (clamped to 0 minimum)
    return max(0.0, prediction[0][0])

if __name__ == "__main__":
    target_player = input("Enter a player name you would like to predict the points for (ensure proper spelling and capitalization): ")
    # Change base_season for testing
    # Use 2026 for actual next-year predictions
    base_season = 2026 
    
    print(f"Running inference for {target_player} using {base_season} metrics...")
    result = predict_player_from_csv(target_player, base_season)
    
    if isinstance(result, str):
        print(result)
    else:
        print("\n" + "="*50)
        print(f"Projected Points for {base_season}-{base_season+1} Season: {result:.1f}")
        print("="*50)