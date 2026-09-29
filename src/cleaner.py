import pandas as pd
import numpy as np
import os

try:
    from src.loader import load_data
except ModuleNotFoundError:
    from loader import load_data

def consolidate_player_transfers(df):
    """Consolidate multiple team rows for players transferred mid-season."""
    if 'Player' not in df.columns:
        return df

    group_key = 'player_id' if 'player_id' in df.columns else 'Player'
    
    # Identify numeric columns for summation vs text columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = [c for c in df.columns if c not in numeric_cols and c != group_key]

    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = 'sum'
    for col in text_cols:
        agg_dict[col] = 'last'

    consolidated = df.groupby(group_key, as_index=False).agg(agg_dict)
    return consolidated

def clean_data(df, is_player_dataset=False):
    if df is None or df.empty:
        return None
    
    # Standardize player_id column
    if '-9999' in df.columns:
        df = df.rename(columns={'-9999': 'player_id'})

    # Drop FBref unwanted columns & repeated header rows
    df = df.drop(columns=['Matches', 'Notes', 'Match Report', 'Rk'], errors='ignore')
    
    if 'Player' in df.columns:
        df = df[df['Player'] != 'Player'].copy()

    if 'Nation' in df.columns:
        df['Nation'] = df['Nation'].astype(str).str.split().str[-1]

    # Fill NaNs in text/numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Consolidate transfers if player dataset
    if is_player_dataset and ('player_id' in df.columns or 'Player' in df.columns):
        df = consolidate_player_transfers(df)

    df = df.reset_index(drop=True)
    return df

if __name__ == "__main__":
    paths = [
        "data/raw/premier_league_2025_2026/matches.csv",
        "data/raw/premier_league_2025_2026/goalkeeping.csv",
        "data/raw/premier_league_2025_2026/miscellaneous_stats.csv",
        "data/raw/premier_league_2025_2026/players.csv",
        "data/raw/premier_league_2025_2026/playing_time.csv",
        "data/raw/premier_league_2025_2026/shooting.csv",
        "data/raw/premier_league_2025_2026/teams.csv",
    ]
    
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    for path in paths:
        raw_df = load_data(path)
        if raw_df is not None:
            is_player = any(fn in path for fn in ["players.csv", "shooting.csv", "miscellaneous_stats.csv", "goalkeeping.csv", "playing_time.csv"])
            clean_df = clean_data(raw_df, is_player_dataset=is_player)
            file_name = os.path.basename(path)
            save_path = os.path.join(processed_dir, file_name)
            clean_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"Saved {save_path} (Shape: {clean_df.shape})")
            print("-" * 50)
