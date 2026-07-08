import pandas as pd
import tensorflow as tf
import os
import time
import pickle
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nhl_webscraper import scrape_team_roster_stats

# Assuming you modified scrape_team_roster_stats to end with:
# return df
# Instead of: df.to_csv(...)

csv_target = "nhl_historical_stats.csv"

# 1. Correct file existence check
if not os.path.isfile(csv_target):
    
    # 2. Corrected abbreviations (ensure all 32 are present)
    all_abbr = ["COL", "DAL", "MIN", "UTA", "STL", "NSH", "WPG", "CHI", "VEG", "EDM", "ANA", "LAK", "SJS", "SEA", "CGY", "VAN", "BUF", "TBL", "MTL", "BOS", "OTT", "DET", "FLA", "TOR", "CAR", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH", "CBJ"]
    
    # Create an empty list to hold the dataframes in memory
    all_dataframes = [] 
    
    for yr in range(2010, 2027):
        for team in all_abbr:
            print(f"Scraping {team} for {yr}...")
            
            # Fetch the dataframe for this specific team and year
            df = scrape_team_roster_stats(yr, team)

            if df is not None and not df.empty:
                # IMPORTANT: Tag the data with the Year and Team before merging it!
                # Otherwise, you won't know who played when in your master file.
                df['Season'] = yr
                df['Team'] = team
                
                # Add it to our storage list
                all_dataframes.append(df)
            
            # 3. Sleep for 3 seconds between requests to avoid getting IP banned
            time.sleep(3) 

    # 4. Once all loops are done, combine everything and save it ONCE
    if all_dataframes:
        print("Concatenating all data...")
        master_df = pd.concat(all_dataframes, ignore_index=True)
        master_df.to_csv(csv_target, index=False)
        print(f"Master CSV created successfully with {len(master_df)} rows!")
    else:
        print("No data was scraped.")
else:
    print(f"{csv_target} already exists. Skipping scrape.")

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

def load_and_preprocess_data(csv_filepath):
    df = pd.read_csv(csv_filepath)
    print(f"Initial rows loaded from CSV: {len(df)}")
    
    # 1. Clean time strings
    time_cols = ['TOI', 'ATOI']
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(time_to_seconds)
            
    # 2. One-Hot Encode position
    if 'Pos' in df.columns:
        df = pd.get_dummies(df, columns=['Pos'], dtype=int)
        
    # 3. Create target variable
    df = df.sort_values(by=['Player', 'Season'])
    df['Next_Season_PTS'] = df.groupby('Player')['PTS'].shift(-1)
    
    df = df.dropna(subset=['Next_Season_PTS'])
    
    # Define features
    exclude_cols = ['Player', 'Team', 'Season', 'Next_Season_PTS', 'Awards', 'xG', 'CF%', 'FF%', "+/-"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    df = df.dropna(subset=feature_cols)
    
    if len(df) == 0:
        raise ValueError("Dataset is empty. Check the warnings above to see where data was lost.")
        
    X = df[feature_cols].values
    y = df['Next_Season_PTS'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, len(feature_cols), scaler
def build_model(input_dim):
    """
    Compiles a Dense Neural Network optimized for continuous regression.
    """
    model = keras.Sequential([
        # Input layer matching the number of features
        keras.layers.Input(shape=(input_dim,)),
        
        # Hidden Layer 1
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),  # Helps prevent overfitting
        
        # Hidden Layer 2
        keras.layers.Dense(32, activation='relu'),
        
        # Output Layer: Single neuron with linear activation for continuous values
        keras.layers.Dense(1, activation='linear')
    ])
    
    # Compile with Adam optimizer and Mean Squared Error loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']  # Mean Absolute Error tells us average point deviation
    )
    return model

if __name__ == "__main__":
    csv_file = "nhl_historical_stats.csv"
    
    try:
        # FIX: Unpack the 'scaler' variable at the end of the line
        X_train, X_test, y_train, y_test, num_features, scaler = load_and_preprocess_data(csv_file)
        
        # Now 'scaler' is defined in this scope, so this line will work perfectly!
        print("Saving feature scaler to scaler.pkl...")
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
            
        # 2. Design network
        nhl_model = build_model(num_features)
        nhl_model.summary()
        
        # 3. Train model
        print("\nStarting model training...")
        history = nhl_model.fit(
            X_train, y_train,
            validation_split=0.15,
            # Epochs set to 500 for training recent models
            epochs=500,
            batch_size=32,
            verbose=1
        )
        
        # 4. Evaluate performance
        print("\nEvaluating model on test set...")
        test_loss, test_mae = nhl_model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Mean Absolute Error: {test_mae:.2f} points")
        
        # 5. Save model
        nhl_model.save("nhl_point_predictor.keras")
        print("Model saved successfully as nhl_point_predictor.keras")
        
    except FileNotFoundError:
        print(f"Could not find {csv_file}.")
    
class TextColor:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

