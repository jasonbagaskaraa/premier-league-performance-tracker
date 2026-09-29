import os
import pandas as pd
from src.loader import load_data
from src.cleaner import clean_data
from src.metrics import (
    calculate_shooting_metrics,
    calculate_player_metrics,
    calculate_discipline_metrics,
    calculate_gk_metrics
)
from src.scoring import calculate_composite_scores
from src.trend import calculate_team_form
from src.predict import train_player_prediction_model, train_match_prediction_model
from src.visualize import (
    plot_top_scorers,
    plot_shooting_efficiency,
    plot_discipline_ranking,
    plot_gk_performance,
    plot_team_goals,
    plot_composite_rankings,
    plot_position_rankings,
    plot_team_form_trend
)

def run_pipeline():
    print("Starting Premier League Performance Tracker Pipeline...")
    
    raw_dir = os.path.join("data", "raw", "premier_league_2025_2026")
    processed_dir = os.path.join("data", "processed")
    metrics_dir = os.path.join("data", "metrics")
    output_dir = "outputs"
    
    for folder in [processed_dir, metrics_dir, output_dir]:
        os.makedirs(folder, exist_ok=True)

    # Step 1: Clean Raw Data
    print("\n[1/4] Cleaning raw data...")
    raw_files = [
        "matches.csv",
        "goalkeeping.csv",
        "miscellaneous_stats.csv",
        "players.csv",
        "playing_time.csv",
        "shooting.csv",
        "teams.csv"
    ]
    
    for filename in raw_files:
        filepath = os.path.join(raw_dir, filename)
        if os.path.exists(filepath):
            df = load_data(filepath)
            cleaned_df = clean_data(df)
            if cleaned_df is not None:
                save_path = os.path.join(processed_dir, filename)
                cleaned_df.to_csv(save_path, index=False)
                print(f"  Processed: {filename} -> {save_path}")

    # Step 2: Calculate Metrics
    print("\n[2/4] Calculating derived metrics...")
    
    shooting_path = os.path.join(processed_dir, "shooting.csv")
    if os.path.exists(shooting_path):
        shooting_df = calculate_shooting_metrics(pd.read_csv(shooting_path))
        shooting_df.to_csv(os.path.join(metrics_dir, "shooting_metrics.csv"), index=False)
        print("  Calculated: shooting_metrics.csv")

    players_path = os.path.join(processed_dir, "players.csv")
    if os.path.exists(players_path):
        players_df = calculate_player_metrics(pd.read_csv(players_path))
        players_df.to_csv(os.path.join(metrics_dir, "player_metrics.csv"), index=False)
        print("  Calculated: player_metrics.csv")

    misc_path = os.path.join(processed_dir, "miscellaneous_stats.csv")
    if os.path.exists(misc_path):
        misc_df = calculate_discipline_metrics(pd.read_csv(misc_path))
        misc_df.to_csv(os.path.join(metrics_dir, "discipline_metrics.csv"), index=False)
        print("  Calculated: discipline_metrics.csv")

    gk_path = os.path.join(processed_dir, "goalkeeping.csv")
    if os.path.exists(gk_path):
        gk_df = calculate_gk_metrics(pd.read_csv(gk_path))
        gk_df.to_csv(os.path.join(metrics_dir, "gk_metrics.csv"), index=False)
        print("  Calculated: gk_metrics.csv")

    scored_df = calculate_composite_scores(metrics_dir)
    scored_df.to_csv(os.path.join(metrics_dir, "composite_scores.csv"), index=False)
    print("  Calculated: composite_scores.csv")

    form_df = calculate_team_form(processed_dir, metrics_dir)
    print("  Calculated: team_form_trend.csv")

    # Step 3: Generate Visualizations
    print("\n[3/4] Generating visualizations...")
    
    shooting_metrics_df = pd.read_csv(os.path.join(metrics_dir, "shooting_metrics.csv"))
    player_metrics_df = pd.read_csv(os.path.join(metrics_dir, "player_metrics.csv"))
    discipline_metrics_df = pd.read_csv(os.path.join(metrics_dir, "discipline_metrics.csv"))
    gk_metrics_df = pd.read_csv(os.path.join(metrics_dir, "gk_metrics.csv"))
    composite_metrics_df = pd.read_csv(os.path.join(metrics_dir, "composite_scores.csv"))
    team_form_df = pd.read_csv(os.path.join(metrics_dir, "team_form_trend.csv"))
    teams_df = pd.read_csv(os.path.join(processed_dir, "teams.csv"))
    
    plot_top_scorers(player_metrics_df, output_dir)
    plot_shooting_efficiency(shooting_metrics_df, output_dir)
    plot_discipline_ranking(discipline_metrics_df, output_dir)
    plot_gk_performance(gk_metrics_df, output_dir)
    plot_team_goals(teams_df, output_dir)
    plot_composite_rankings(composite_metrics_df, output_dir)
    plot_position_rankings(composite_metrics_df, output_dir)
    plot_team_form_trend(team_form_df, output_dir)

    # Step 4: Machine Learning Predictions
    print("\n[4/4] Training machine learning prediction models...")
    train_player_prediction_model(metrics_dir, output_dir)
    train_match_prediction_model(processed_dir, metrics_dir, output_dir)

    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()
