"""
Premier League Performance Tracker — Flask Web Dashboard
Serves CSV metrics as JSON API endpoints and hosts the interactive frontend.
"""
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from src.predict import predict_fixture, train_match_prediction_model

app = Flask(__name__, static_folder="dashboard", static_url_path="")
CORS(app)

METRICS_DIR = os.path.join("data", "metrics")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUTS_DIR = "outputs"

# Lazy-loaded match model & history cache
_match_model_cache = {}

def _read_csv(path, **kwargs):
    """Read CSV and clean NaN for JSON serialization."""
    df = pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    return df.where(pd.notnull(df), None)

def _df_to_records(df):
    """Convert DataFrame to list of dicts with JSON-safe values."""
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                rec[k] = None
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v)
    return records

# ─── Static Frontend ───────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")

# ─── API Endpoints ─────────────────────────────────────────────
@app.route("/api/teams")
def api_teams():
    df = _read_csv(os.path.join(PROCESSED_DIR, "teams.csv"))
    return jsonify(_df_to_records(df))

@app.route("/api/team-form")
def api_team_form():
    df = _read_csv(os.path.join(METRICS_DIR, "team_form_trend.csv"))
    return jsonify(_df_to_records(df))

@app.route("/api/composite-scores")
def api_composite_scores():
    df = _read_csv(os.path.join(METRICS_DIR, "composite_scores.csv"))
    # Return top 50 + filter options
    top = df.dropna(subset=["composite_score"]).sort_values("composite_score", ascending=False).head(50)
    return jsonify({
        "players": _df_to_records(top),
        "positions": sorted(df["pos_group"].dropna().unique().tolist()),
        "squads": sorted(df["Squad"].dropna().unique().tolist())
    })

@app.route("/api/player-predictions")
def api_player_predictions():
    df = _read_csv(os.path.join(METRICS_DIR, "player_ml_predictions.csv"))
    return jsonify(_df_to_records(df))

@app.route("/api/match-predictions")
def api_match_predictions():
    df = _read_csv(os.path.join(METRICS_DIR, "match_ml_predictions.csv"))
    # Return last 40 matches
    tail = df.tail(40)
    return jsonify(_df_to_records(tail))

@app.route("/api/shooting-metrics")
def api_shooting_metrics():
    df = _read_csv(os.path.join(METRICS_DIR, "shooting_metrics.csv"))
    top = df.dropna(subset=["Gls"]).sort_values("Gls", ascending=False).head(20)
    return jsonify(_df_to_records(top))

@app.route("/api/gk-metrics")
def api_gk_metrics():
    df = _read_csv(os.path.join(METRICS_DIR, "gk_metrics.csv"))
    return jsonify(_df_to_records(df))

@app.route("/api/predict-match", methods=["POST"])
def api_predict_match():
    """Predict a custom fixture between two teams."""
    data = request.get_json()
    home = data.get("home_team", "").strip()
    away = data.get("away_team", "").strip()
    if not home or not away:
        return jsonify({"error": "Both home_team and away_team are required"}), 400

    # Use cached model or train fresh
    if "model" not in _match_model_cache:
        _, model, history = train_match_prediction_model(PROCESSED_DIR, METRICS_DIR, OUTPUTS_DIR)
        _match_model_cache["model"] = model
        _match_model_cache["history"] = history

    result = predict_fixture(
        home, away, PROCESSED_DIR, METRICS_DIR,
        _match_model_cache["model"], _match_model_cache["history"]
    )
    return jsonify(result)

@app.route("/api/charts/<filename>")
def api_chart_image(filename):
    """Serve pre-generated chart images."""
    return send_from_directory(OUTPUTS_DIR, filename)

if __name__ == "__main__":
    print("Starting Premier League Dashboard at http://localhost:5000")
    app.run(debug=False, port=5000)
