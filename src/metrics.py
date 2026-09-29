import pandas as pd
import os

def safe_divide(numerator, denominator, fill=0):
    return (numerator / denominator).fillna(fill).replace([float('inf'), float('-inf')], fill)

def calculate_shooting_metrics(df):
    df = df.copy()
    
    df['Sh'] = pd.to_numeric(df['Sh'], errors='coerce').fillna(0)
    df['SoT'] = pd.to_numeric(df['SoT'], errors='coerce').fillna(0)
    df['Gls'] = pd.to_numeric(df['Gls'], errors='coerce').fillna(0)
    
    df['shot_accuracy'] = (safe_divide(df['SoT'], df['Sh']) * 100).round(1)
    df['goal_conversion'] = (safe_divide(df['Gls'], df['Sh']) * 100).round(1)
    
    return df

def calculate_player_metrics(df):
    df = df.copy()
    
    df['Min'] = pd.to_numeric(df['Min'], errors='coerce').fillna(0)
    df['Gls'] = pd.to_numeric(df['Gls'], errors='coerce').fillna(0)
    df['Ast'] = pd.to_numeric(df['Ast'], errors='coerce').fillna(0)
    
    goal_contributions = df['Gls'] + df['Ast']
    df['mins_per_goal_contribution'] = safe_divide(df['Min'], goal_contributions).round(1)
    df['goal_involvement_per90'] = safe_divide(goal_contributions * 90, df['Min']).round(2)
    
    return df

def calculate_discipline_metrics(df):
    df = df.copy()
    
    df['CrdY'] = pd.to_numeric(df['CrdY'], errors='coerce').fillna(0)
    df['CrdR'] = pd.to_numeric(df['CrdR'], errors='coerce').fillna(0)
    df['Fls'] = pd.to_numeric(df['Fls'], errors='coerce').fillna(0)
    df['Fld'] = pd.to_numeric(df['Fld'], errors='coerce').fillna(0)
    df['90s'] = pd.to_numeric(df['90s'], errors='coerce').fillna(0)
    
    df['fouls_per90'] = safe_divide(df['Fls'], df['90s']).round(2)
    df['fouled_per90'] = safe_divide(df['Fld'], df['90s']).round(2)
    df['discipline_points'] = df['CrdY'] + (df['CrdR'] * 3)
    
    return df

def calculate_gk_metrics(df):
    df = df.copy()
    
    df['GA'] = pd.to_numeric(df['GA'], errors='coerce').fillna(0)
    df['SoTA'] = pd.to_numeric(df['SoTA'], errors='coerce').fillna(0)
    df['Saves'] = pd.to_numeric(df['Saves'], errors='coerce').fillna(0)
    df['CS'] = pd.to_numeric(df['CS'], errors='coerce').fillna(0)
    df['MP'] = pd.to_numeric(df['MP'], errors='coerce').fillna(0)
    
    df['save_pct'] = (safe_divide(df['Saves'], df['SoTA']) * 100).round(1)
    df['cs_pct'] = (safe_divide(df['CS'], df['MP']) * 100).round(1)
    
    return df

if __name__ == "__main__":
    processed_dir = os.path.join("data", "processed")
    metrics_dir = os.path.join("data", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    shooting_df = pd.read_csv(os.path.join(processed_dir, "shooting.csv"), encoding='utf-8-sig')
    shooting_df = calculate_shooting_metrics(shooting_df)
    shooting_df.to_csv(os.path.join(metrics_dir, "shooting_metrics.csv"), index=False, encoding='utf-8-sig')
    
    players_df = pd.read_csv(os.path.join(processed_dir, "players.csv"), encoding='utf-8-sig')
    players_df = calculate_player_metrics(players_df)
    players_df.to_csv(os.path.join(metrics_dir, "player_metrics.csv"), index=False, encoding='utf-8-sig')
    
    misc_df = pd.read_csv(os.path.join(processed_dir, "miscellaneous_stats.csv"), encoding='utf-8-sig')
    misc_df = calculate_discipline_metrics(misc_df)
    misc_df.to_csv(os.path.join(metrics_dir, "discipline_metrics.csv"), index=False, encoding='utf-8-sig')
    
    gk_df = pd.read_csv(os.path.join(processed_dir, "goalkeeping.csv"), encoding='utf-8-sig')
    gk_df = calculate_gk_metrics(gk_df)
    gk_df.to_csv(os.path.join(metrics_dir, "gk_metrics.csv"), index=False, encoding='utf-8-sig')
    print("Metrics calculated and saved with UTF-8-SIG encoding.")
