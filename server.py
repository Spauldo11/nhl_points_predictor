import os
import sys
import json
import pickle
import urllib.parse
import pandas as pd
import numpy as np

# Ensure TensorFlow / Keras suppresses excessive logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow/Keras not found in runtime.")

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILEPATH = os.path.join(BASE_DIR, "nhl_historical_stats.csv")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

MODEL_PATHS = {
    "m1": os.path.join(BASE_DIR, "nhl_point_predictor.keras"),
    "m2": os.path.join(BASE_DIR, "nhl_point_predictorv2.keras"),
    "m3": os.path.join(BASE_DIR, "nhl_points_predictorv3.keras"),
    "m4": os.path.join(BASE_DIR, "nhl_points_predictorv4.keras"),
}

# Global loaded artifacts
SCALER = None
MODELS = {}
DATASET_DF = None
PREPROCESSED_DF = None
FEATURE_COLS = []
AVAILABLE_PLAYERS = []
AVAILABLE_SEASONS = []

def time_to_seconds(time_str):
    """Converts a 'MM:SS' time string into a float of total seconds."""
    if pd.isna(time_str) or not isinstance(time_str, str):
        return 0.0
    if ':' in time_str:
        parts = time_str.split(':')
        try:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return float((minutes * 60) + seconds)
        except ValueError:
            return 0.0
    try:
        return float(time_str)
    except ValueError:
        return 0.0

def load_artifacts():
    global SCALER, MODELS, DATASET_DF, PREPROCESSED_DF, FEATURE_COLS, AVAILABLE_PLAYERS, AVAILABLE_SEASONS

    if PREPROCESSED_DF is not None and SCALER is not None:
        return True

    print("Loading dataset...")
    if not os.path.exists(CSV_FILEPATH):
        print(f"Error: Dataset {CSV_FILEPATH} not found.")
        return False

    DATASET_DF = pd.read_csv(CSV_FILEPATH)
    
    # Extract unique player list & seasons
    if 'Player' in DATASET_DF.columns:
        AVAILABLE_PLAYERS = sorted(list(DATASET_DF['Player'].dropna().unique()))
    if 'Season' in DATASET_DF.columns:
        AVAILABLE_SEASONS = sorted([int(s) for s in DATASET_DF['Season'].dropna().unique()], reverse=True)

    # Replicate preprocessing
    df_clean = DATASET_DF.copy()
    time_cols = ['TOI', 'ATOI']
    for col in time_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).apply(time_to_seconds)

    if 'Pos' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['Pos'], dtype=int)

    exclude_cols = ['Player', 'Team', 'Season', 'Next_Season_PTS', 'Awards', 'xG', 'CF%', 'FF%']
    FEATURE_COLS = [col for col in df_clean.columns if col not in exclude_cols]

    for col in FEATURE_COLS:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    PREPROCESSED_DF = df_clean

    # Load scaler
    print("Loading scaler...")
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, "rb") as f:
                SCALER = pickle.load(f)
            print("Scaler loaded successfully.")
        except Exception as e:
            print(f"Failed to load scaler: {e}")
    else:
        print(f"Warning: {SCALER_PATH} not found.")

    # Load Keras models
    print("Loading Keras models...")
    if HAS_TF:
        for key, path in MODEL_PATHS.items():
            if os.path.exists(path):
                try:
                    MODELS[key] = keras.models.load_model(path)
                    print(f"Loaded model {key} from {os.path.basename(path)}")
                except Exception as e:
                    print(f"Error loading model {key}: {e}")
            else:
                print(f"Model file missing: {path}")

    print("Artifact loading complete.\n")
    return True

# Auto-load artifacts on module import
load_artifacts()

def run_prediction(player_name, season=2026):
    if PREPROCESSED_DF is None or SCALER is None:
        load_artifacts()
    if PREPROCESSED_DF is None or SCALER is None:
        return {"error": "Server models or dataset not loaded properly."}

    season = int(season)
    
    # Perform case-insensitive matching
    df_raw = DATASET_DF
    df_proc = PREPROCESSED_DF

    # Match player name
    matched_players = df_raw[df_raw['Player'].str.lower() == player_name.strip().lower()]['Player'].unique()
    
    if len(matched_players) == 0:
        # Partial match fallback
        matched_players = df_raw[df_raw['Player'].str.lower().str.contains(player_name.strip().lower(), na=False)]['Player'].unique()
        if len(matched_players) == 0:
            return {"error": f"Could not find stats for player '{player_name}'."}

    exact_player_name = matched_players[0]
    
    # Get available seasons for player
    player_seasons = sorted([int(s) for s in df_raw[df_raw['Player'] == exact_player_name]['Season'].unique()], reverse=True)

    # Check if target season exists
    player_data = df_proc[(df_proc['Player'] == exact_player_name) & (df_proc['Season'] == season)]
    
    selected_season = season
    season_fallback_warning = None
    if player_data.empty:
        if len(player_seasons) > 0:
            selected_season = player_seasons[0]
            player_data = df_proc[(df_proc['Player'] == exact_player_name) & (df_proc['Season'] == selected_season)]
            season_fallback_warning = f"No stats found for {exact_player_name} in {season}. Used latest available season ({selected_season})."
        else:
            return {"error": f"No historical stats found for {exact_player_name}."}

    # Handle TOT row for mid-season trades
    if len(player_data) > 1:
        if 'Team' in player_data.columns and 'TOT' in player_data['Team'].values:
            player_data = player_data[player_data['Team'] == 'TOT']
        else:
            player_data = player_data.iloc[[0]]

    # Raw row from dataset for metadata display
    raw_row = df_raw[(df_raw['Player'] == exact_player_name) & (df_raw['Season'] == selected_season)].iloc[0].to_dict()
    for k, v in raw_row.items():
        if pd.isna(v):
            raw_row[k] = None

    # Process features array
    player_row = player_data[FEATURE_COLS].copy()

    # Impute missing feature values
    imputed_cols = []
    for col in FEATURE_COLS:
        if player_row[col].isna().iloc[0]:
            imputed_cols.append(col)
            if col in ['CF%', 'FF%', 'oiSH%', 'oZS%', 'xG']:
                league_avg = df_proc[col].mean()
                player_row[col] = player_row[col].fillna(league_avg)
            else:
                player_row[col] = player_row[col].fillna(0.0)

    stats_array = player_row.values
    scaled_stats = SCALER.transform(stats_array)

    predictions = {}
    model_labels = {
        "m1": "100 Epochs (Primary)",
        "m2": "500 Epochs",
        "m3": "500 Epochs (No +/-)",
        "m4": "500 Epochs (Reduced +/-)"
    }

    primary_prediction = 0.0
    valid_pts = []
    valid_raw = []

    for key, model in MODELS.items():
        try:
            pred_val = float(model.predict(scaled_stats, verbose=0)[0][0])
            pred_clamped = max(0.0, pred_val)
            predictions[key] = {
                "label": model_labels.get(key, key),
                "pts": round(pred_clamped, 1),
                "raw_pts": round(pred_val, 2)
            }
            valid_pts.append(pred_clamped)
            valid_raw.append(pred_val)
            if key == "m1":
                primary_prediction = round(pred_clamped, 1)
        except Exception as e:
            predictions[key] = {"label": model_labels.get(key, key), "error": str(e)}

    # Calculate Consensus Average of all models
    if valid_pts:
        avg_clamped = round(sum(valid_pts) / len(valid_pts), 1)
        avg_raw = round(sum(valid_raw) / len(valid_raw), 2)
        predictions["avg"] = {
            "label": "Consensus Average (All Models)",
            "pts": avg_clamped,
            "raw_pts": avg_raw
        }
    else:
        predictions["avg"] = {
            "label": "Consensus Average (All Models)",
            "pts": primary_prediction,
            "raw_pts": primary_prediction
        }

    # Feature breakdown dictionary for UI display
    feature_dict = {}
    for col in FEATURE_COLS:
        val = player_row[col].iloc[0]
        feature_dict[col] = round(float(val), 2) if isinstance(val, (int, float, np.number)) else str(val)

    return {
        "player_name": exact_player_name,
        "target_season": season,
        "used_season": selected_season,
        "warning": season_fallback_warning,
        "available_seasons": player_seasons,
        "primary_prediction": primary_prediction,
        "model_predictions": predictions,
        "raw_stats": raw_row,
        "features": feature_dict,
        "imputed_features": imputed_cols
    }

# Try Flask first, fallback to HTTP Server
try:
    # pyrefly: ignore [missing-import]
    from flask import Flask, request, jsonify, send_from_directory
    try:
        from flask_cors import CORS
        HAS_CORS = True
    except ImportError:
        HAS_CORS = False

    app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
    if HAS_CORS:
        CORS(app)

    @app.route('/')
    def serve_index():
        return send_from_directory(BASE_DIR, 'index.html')

    @app.route('/<path:path>')
    def serve_static(path):
        if os.path.exists(os.path.join(BASE_DIR, path)):
            return send_from_directory(BASE_DIR, path)
        return send_from_directory(BASE_DIR, 'index.html')

    @app.route('/api/predict', methods=['GET'])
    def api_predict():
        player = request.args.get('player', '')
        season = request.args.get('season', 2026)
        if not player:
            return jsonify({"error": "Missing 'player' parameter"}), 400
        res = run_prediction(player, season)
        if "error" in res:
            return jsonify(res), 404
        return jsonify(res)

    @app.route('/api/players', methods=['GET'])
    def api_players():
        query = request.args.get('query', '').strip().lower()
        limit = int(request.args.get('limit', 15))
        if query:
            matches = [p for p in AVAILABLE_PLAYERS if query in p.lower()]
        else:
            matches = AVAILABLE_PLAYERS[:limit]
        return jsonify({"players": matches[:limit], "total": len(matches)})

    @app.route('/api/seasons', methods=['GET'])
    def api_seasons():
        return jsonify({"seasons": AVAILABLE_SEASONS})

    def run_server(port=5000):
        print(f"🚀 NHL Points Predictor Web Server starting on http://0.0.0.0:{port}")
        app.run(host='0.0.0.0', port=port, debug=False)

except ImportError:
    # Standard library fallback server
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class RequestHandler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = urllib.parse.parse_qs(parsed.query)

            if path == '/api/predict':
                player = params.get('player', [''])[0]
                season = params.get('season', ['2026'])[0]
                if not player:
                    self.send_json({"error": "Missing player parameter"}, status=400)
                    return
                res = run_prediction(player, season)
                status = 404 if "error" in res else 200
                self.send_json(res, status=status)

            elif path == '/api/players':
                query = params.get('query', [''])[0].strip().lower()
                limit = int(params.get('limit', ['15'])[0])
                if query:
                    matches = [p for p in AVAILABLE_PLAYERS if query in p.lower()]
                else:
                    matches = AVAILABLE_PLAYERS[:limit]
                self.send_json({"players": matches[:limit], "total": len(matches)})

            elif path == '/api/seasons':
                self.send_json({"seasons": AVAILABLE_SEASONS})

            else:
                filename = 'index.html' if path in ('/', '') else path.lstrip('/')
                filepath = os.path.join(BASE_DIR, filename)
                if os.path.exists(filepath) and os.path.isfile(filepath):
                    ext = os.path.splitext(filepath)[1].lower()
                    mime_types = {
                        '.html': 'text/html',
                        '.css': 'text/css',
                        '.js': 'application/javascript',
                        '.json': 'application/json',
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.svg': 'image/svg+xml',
                        '.ico': 'image/x-icon'
                    }
                    content_type = mime_types.get(ext, 'application/octet-stream')
                    with open(filepath, 'rb') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', content_type)
                    self.send_header('Content-Length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self.send_error(404, "File Not Found")

        def send_json(self, data, status=200):
            body = json.dumps(data).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]} {args[1]}\n")

    def run_server(port=5000):
        HTTPServer.allow_reuse_address = True
        server_address = ('0.0.0.0', port)
        httpd = HTTPServer(server_address, RequestHandler)
        print(f"NHL Points Predictor Web Server starting on http://0.0.0.0:{port}")
        httpd.serve_forever()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    run_server(port)
