import pandas as pd
import numpy as np
import os

def calc_percentile_rank(series, invert=False):
    """Calculates percentile rank (0-100) within positional peer group."""
    if len(series) <= 1:
        return pd.Series(50.0, index=series.index)
    ranks = series.rank(pct=True) * 100.0
    if invert:
        return 100.0 - ranks
    return ranks

def aggregate_player_records(df, num_cols):
    """Aggregates stats for players with multiple team entries in a single season."""
    if 'Player' not in df.columns:
        return df
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    group_col = 'player_id' if 'player_id' in df.columns else 'Player'
    agg_dict = {col: 'sum' for col in num_cols if col in df.columns}
    for c in ['Player', 'Pos', 'Nation', 'Squad']:
        if c in df.columns:
            agg_dict[c] = 'last'
            
    return df.groupby(group_col).agg(agg_dict).reset_index(drop=True)

def classify_position(pos_str):
    pos = str(pos_str).upper()
    if 'GK' in pos:
        return 'GK'
    elif 'DF' in pos:
        return 'DF'
    elif 'MF' in pos:
        if 'FW' in pos or 'AM' in pos or 'WING' in pos:
            return 'AM_WG'
        return 'CM_DM'
    elif 'FW' in pos:
        return 'FW'
    return 'CM_DM'

def calculate_composite_scores(metrics_dir):
    """
    Calculate positional percentile performance scores (0-100 scale)
    for DF, CM_DM, AM_WG, FW, and GK using volume, rate, and sample thresholding.
    """
    processed_dir = os.path.join("data", "processed")
    players_path = os.path.join(processed_dir, "players.csv")
    shooting_path = os.path.join(processed_dir, "shooting.csv")
    discipline_path = os.path.join(processed_dir, "miscellaneous_stats.csv")
    gk_path = os.path.join(processed_dir, "goalkeeping.csv")

    players_df = pd.read_csv(players_path, encoding='utf-8-sig')
    shooting_df = pd.read_csv(shooting_path, encoding='utf-8-sig')
    misc_df = pd.read_csv(discipline_path, encoding='utf-8-sig')
    gk_df = pd.read_csv(gk_path, encoding='utf-8-sig') if os.path.exists(gk_path) else pd.DataFrame()

    clean_players = aggregate_player_records(players_df, ['Min', '90s', 'Gls', 'Ast', 'G+A', 'CrdY', 'CrdR'])
    clean_shooting = aggregate_player_records(shooting_df, ['Sh', 'SoT', 'Gls', 'PK', 'PKatt'])
    clean_misc = aggregate_player_records(misc_df, ['CrdY', 'CrdR', 'Fls', 'Fld', 'Int', 'TklW', 'OG'])

    # Team Defense Calculation from GK data
    if not gk_df.empty:
        gk_df['GA'] = pd.to_numeric(gk_df['GA'], errors='coerce').fillna(0)
        gk_df['CS'] = pd.to_numeric(gk_df['CS'], errors='coerce').fillna(0)
        gk_df['90s'] = pd.to_numeric(gk_df['90s'], errors='coerce').fillna(0)
        team_gk = gk_df.groupby('Squad').agg({'GA': 'sum', '90s': 'max', 'CS': 'sum'}).reset_index()
        team_gk['team_ga90'] = (team_gk['GA'] / team_gk['90s'].replace(0, np.nan)).fillna(1.5)
        team_gk['team_cs_pct'] = ((team_gk['CS'] / team_gk['90s'].replace(0, np.nan)).fillna(0)) * 100
    else:
        team_gk = pd.DataFrame({'Squad': [], 'team_ga90': [], 'team_cs_pct': []})

    df = pd.merge(clean_players, clean_shooting[['Player', 'Sh', 'SoT']], on='Player', how='left')
    df = pd.merge(df, clean_misc[['Player', 'Fls', 'Fld', 'Int', 'TklW']], on='Player', how='left').fillna(0)
    df = pd.merge(df, team_gk[['Squad', 'team_ga90', 'team_cs_pct']], on='Squad', how='left')

    df['90s'] = pd.to_numeric(df['90s'], errors='coerce').replace(0, np.nan)
    df['ga_total'] = df['Gls'] + df['Ast']
    df['ga_per90'] = (df['ga_total'] / df['90s']).fillna(0)
    df['tkl_int_total'] = df['TklW'] + df['Int']
    df['tkl_int_per90'] = (df['tkl_int_total'] / df['90s']).fillna(0)
    df['discipline_points'] = df['CrdY'] + (df['CrdR'] * 3)

    # Minimum playing time threshold: 10.0 90s (~900 mins / 10 full matches)
    qualified = df[df['90s'] >= 10.0].copy()
    qualified['pos_group'] = qualified['Pos'].apply(classify_position)

    scored_chunks = []

    for pgroup, group in qualified.groupby('pos_group'):
        g = group.copy()
        
        if pgroup == 'DF':
            g['composite_score'] = (
                (calc_percentile_rank(g['team_ga90'], invert=True) * 0.35) +
                (calc_percentile_rank(g['team_cs_pct']) * 0.25) +
                (calc_percentile_rank(g['tkl_int_total']) * 0.20) +
                (calc_percentile_rank(g['90s']) * 0.20)
            )
        elif pgroup == 'CM_DM':
            g['composite_score'] = (
                (calc_percentile_rank(g['tkl_int_total']) * 0.30) +
                (calc_percentile_rank(g['ga_total']) * 0.25) +
                (calc_percentile_rank(g['ga_per90']) * 0.25) +
                (calc_percentile_rank(g['90s']) * 0.20)
            )
        elif pgroup == 'AM_WG':
            g['composite_score'] = (
                (calc_percentile_rank(g['ga_total']) * 0.40) +
                (calc_percentile_rank(g['ga_per90']) * 0.35) +
                (calc_percentile_rank(g['90s']) * 0.25)
            )
        elif pgroup == 'FW':
            g['composite_score'] = (
                (calc_percentile_rank(g['ga_total']) * 0.40) +
                (calc_percentile_rank(g['ga_per90']) * 0.35) +
                (calc_percentile_rank(g['90s']) * 0.25)
            )

        scored_chunks.append(g)

    # Goalkeepers: min 10 matches
    if not gk_df.empty:
        gk = gk_df.copy()
        gk['MP'] = pd.to_numeric(gk['MP'], errors='coerce').fillna(0)
        gk_qual = gk[gk['MP'] >= 10].copy()
        
        save_col = 'Save%' if 'Save%' in gk_qual.columns else 'save_pct'
        cs_col = 'CS%' if 'CS%' in gk_qual.columns else 'cs_pct'
        
        save_vals = pd.to_numeric(gk_qual[save_col], errors='coerce').fillna(0)
        cs_vals = pd.to_numeric(gk_qual[cs_col], errors='coerce').fillna(0)
        ga90_vals = pd.to_numeric(gk_qual['GA90'], errors='coerce').fillna(0)

        p_save = calc_percentile_rank(save_vals)
        p_cs = calc_percentile_rank(cs_vals)
        p_ga90 = calc_percentile_rank(ga90_vals, invert=True)

        gk_qual['composite_score'] = (p_save * 0.40) + (p_cs * 0.35) + (p_ga90 * 0.25)
        gk_qual['pos_group'] = 'GK'

        for col in qualified.columns:
            if col not in gk_qual.columns:
                gk_qual[col] = 0

        scored_chunks.append(gk_qual[qualified.columns])

    final_df = pd.concat(scored_chunks, ignore_index=True)
    final_df['composite_score'] = final_df['composite_score'].round(1)
    final_df = final_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    final_df['rank'] = final_df.index + 1

    return final_df

if __name__ == "__main__":
    metrics_dir = os.path.join("data", "metrics")
    scored_df = calculate_composite_scores(metrics_dir)
    save_path = os.path.join(metrics_dir, "composite_scores.csv")
    scored_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Saved position-aware composite scores: {save_path}\n")

    for pgroup, pname in [('DF', 'Defenders'), ('CM_DM', 'Central / Defensive Midfielders'), ('AM_WG', 'Attacking Midfielders & Wingers'), ('FW', 'Forwards'), ('GK', 'Goalkeepers')]:
        print(f"--- Top 5 {pname} ({pgroup}) ---")
        sub_df = scored_df[scored_df['pos_group'] == pgroup].head(5)
        print(sub_df[['rank', 'Player', 'Squad', '90s', 'composite_score']].to_string(index=False))
        print("-" * 50)
