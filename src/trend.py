import pandas as pd
import numpy as np
import os
import re

def parse_score(score_str):
    """Parse score string like '4–2' or '1-0' into (home_goals, away_goals)."""
    if not isinstance(score_str, str) or pd.isna(score_str):
        return np.nan, np.nan
    # Replace en-dash with standard hyphen
    clean_str = score_str.replace('–', '-').strip()
    match = re.search(r'(\d+)\s*-\s*(\d+)', clean_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return np.nan, np.nan

def calculate_team_form(processed_dir, metrics_dir):
    """
    Calculate 5-match rolling form index (Points, Goals For, Goals Against, GD)
    and classify current form status for all Premier League teams.
    """
    matches_path = os.path.join(processed_dir, "matches.csv")
    if not os.path.exists(matches_path):
        matches_path = os.path.join("data", "raw", "premier_league_2025_2026", "matches.csv")

    if not os.path.exists(matches_path):
        raise FileNotFoundError("matches.csv not found in processed or raw directory.")

    matches_df = pd.read_csv(matches_path, encoding='utf-8-sig')
    
    # Filter rows with valid score
    matches_df = matches_df.dropna(subset=['Score']).copy()
    matches_df = matches_df[matches_df['Score'].str.contains(r'\d', na=False)].copy()

    # Parse goals
    parsed_goals = matches_df['Score'].apply(parse_score)
    matches_df['home_gls'] = [g[0] for g in parsed_goals]
    matches_df['away_gls'] = [g[1] for g in parsed_goals]

    matches_df = matches_df.dropna(subset=['home_gls', 'away_gls']).copy()
    matches_df['Wk'] = pd.to_numeric(matches_df['Wk'], errors='coerce').fillna(0)
    matches_df = matches_df.sort_values(['Wk', 'Date']).reset_index(drop=True)

    # Build team-perspective match log
    team_records = []

    for _, row in matches_df.iterrows():
        wk = row['Wk']
        date = row['Date']
        home = row['Home']
        away = row['Away']
        h_gls = int(row['home_gls'])
        a_gls = int(row['away_gls'])

        # Home team perspective
        if h_gls > a_gls:
            h_pts, a_pts = 3, 0
        elif h_gls == a_gls:
            h_pts, a_pts = 1, 1
        else:
            h_pts, a_pts = 0, 3

        team_records.append({
            'Wk': wk, 'Date': date, 'Squad': home, 'Opponent': away, 'Is_Home': 1,
            'GF': h_gls, 'GA': a_gls, 'GD': h_gls - a_gls, 'Pts': h_pts
        })
        team_records.append({
            'Wk': wk, 'Date': date, 'Squad': away, 'Opponent': home, 'Is_Home': 0,
            'GF': a_gls, 'GA': h_gls, 'GD': a_gls - h_gls, 'Pts': a_pts
        })

    logs_df = pd.DataFrame(team_records)
    logs_df = logs_df.sort_values(['Squad', 'Wk', 'Date']).reset_index(drop=True)

    # Rolling 5-match form metrics per team
    logs_df['pts_last_5'] = logs_df.groupby('Squad')['Pts'].transform(lambda s: s.rolling(5, min_periods=1).sum())
    logs_df['gf_last_5'] = logs_df.groupby('Squad')['GF'].transform(lambda s: s.rolling(5, min_periods=1).sum())
    logs_df['ga_last_5'] = logs_df.groupby('Squad')['GA'].transform(lambda s: s.rolling(5, min_periods=1).sum())
    logs_df['gd_last_5'] = logs_df.groupby('Squad')['GD'].transform(lambda s: s.rolling(5, min_periods=1).sum())

    # Get latest form status per team
    latest_form = logs_df.groupby('Squad').last().reset_index()

    def get_form_status(pts):
        if pts >= 11:
            return 'Hot Form'
        elif pts >= 6:
            return 'Stable Form'
        return 'Cold Form'

    latest_form['form_status'] = latest_form['pts_last_5'].apply(get_form_status)
    latest_form = latest_form.sort_values(['pts_last_5', 'gd_last_5', 'gf_last_5'], ascending=[False, False, False]).reset_index(drop=True)
    latest_form['rank'] = latest_form.index + 1

    cols = ['rank', 'Squad', 'Wk', 'pts_last_5', 'gf_last_5', 'ga_last_5', 'gd_last_5', 'form_status']
    result_df = latest_form[cols].copy()

    save_path = os.path.join(metrics_dir, "team_form_trend.csv")
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    return result_df

if __name__ == "__main__":
    processed_dir = os.path.join("data", "processed")
    metrics_dir = os.path.join("data", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    form_df = calculate_team_form(processed_dir, metrics_dir)
    print("Team Form & Trend Analysis (Last 5 Matches):")
    print(form_df.to_string(index=False))
