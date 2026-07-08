import pandas as pd
import requests
import io

def scrape_team_roster_stats(year, team_abbr):
    """
    Scrapes standard skater statistics for a specific team and season.
    Flattens MultiIndex headers for compatibility with CSV and ML pipelines.
    """
    url = f"https://www.hockey-reference.com/teams/{team_abbr}/{year}.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        
        tables = pd.read_html(io.StringIO(response.text), attrs={'id': 'player_stats'})
        df = tables[0]
        
        # --- FIX: Flatten MultiIndex Columns ---
        # If the columns have multiple levels, drop the top level or combine them
        if isinstance(df.columns, pd.MultiIndex):
            # Hockey-Reference clean columns are usually in the second level (level 1)
            # If the second level has an empty string or 'Unnamed', fall back to level 0
            df.columns = [col[1] if 'Unnamed' not in col[1] else col[0] for col in df.columns]
        
        # Clean the DataFrame using our newly flattened column names
        if 'Player' in df.columns:
            df = df[df['Player'] != 'Player'].dropna(subset=['Player'])
            
            # Additional cleanup: Strip any symbols (like '*' for Hall of Famers/Awards) from player names
            df['Player'] = df['Player'].str.replace('*', '', regex=False)
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Skipping {team_abbr} {year} (Likely didn't exist yet).")
        return None
    except ValueError:
        print(f"Could not find the 'skaters' table for {team_abbr} {year}.")
        return None

# Example Usage: Scrape the 2023-2024 San Jose Sharks
if __name__ == "__main__":
    # Note: For the 2023-2024 season, you pass '2024'
    scrape_team_roster_stats(2024, "SJS", "sjs_2024_stats.csv")