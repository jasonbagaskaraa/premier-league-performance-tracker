import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix

def parse_score(score_str):
    """Parse score string like '4–2' or '1-0' into (home_goals, away_goals)."""
    if not isinstance(score_str, str) or pd.isna(score_str):
        return np.nan, np.nan
    clean_str = score_str.replace('–', '-').strip()
    match = re.search(r'(\d+)\s*-\s*(\d+)', clean_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return np.nan, np.nan

def train_player_prediction_model(metrics_dir, output_dir):
    """
    Train holistic role-specific ML regression models tailored to each functional position:
    1. Attackers (FW, AM_WG):
       - Target: Goal Contributions per 90 (G+A/90)
       - Features: Shots/90, Shots on Target/90, SoT%, Fouls Won/90, Composite Score (Caliber Prior), 90s
    2. Midfielders & Fullbacks/Wingbacks (CM_DM, FB_WB):
       - Target: Two-Way Dynamic Impact Score = (G+A/90 * 1.5) + (Tkl+Int/90) + (Crs/90 * 0.5)
       - Features: Tackles+Int/90, Crosses/90, Shots/90, Fouls Won/90, Composite Score (Caliber Prior), 90s
    3. Center-Backs (CB):
       - Target: Defensive Solidity Index = (Tkl+Int/90) + ((2.5 - Team GA90) * 1.2)
       - Features: Tackles+Int/90, Team GA90, Team Clean Sheet %, Composite Score (Caliber Prior), 90s
    4. Goalkeepers (GK):
       - Target: Save Efficiency (Save%)
       - Features: Shots on Target Faced / 90 (SoTA/90), Clean Sheet %, Composite Score (Caliber Prior), 90s
    """
    comp_path = os.path.join(metrics_dir, "composite_scores.csv")
    if not os.path.exists(comp_path):
        raise FileNotFoundError(f"Missing {comp_path}")

    df = pd.read_csv(comp_path, encoding='utf-8-sig')

    # Prepare features
    df['sh_per90'] = (df['Sh'] / df['90s'].replace(0, np.nan)).fillna(0)
    df['sot_per90'] = (df['SoT'] / df['90s'].replace(0, np.nan)).fillna(0)
    df['sot_pct'] = np.where(df['Sh'] > 0, df['SoT'] / df['Sh'], 0.0)
    df['fld_per90'] = (df['Fld'] / df['90s'].replace(0, np.nan)).fillna(0)
    df['crs_per90'] = (df['Crs'] / df['90s'].replace(0, np.nan)).fillna(0) if 'Crs' in df.columns else 0.0
    df['tkl_int_per90'] = (df['tkl_int_total'] / df['90s'].replace(0, np.nan)).fillna(0)
    df['ga_per90'] = (df['ga_total'] / df['90s'].replace(0, np.nan)).fillna(0)
    df['composite_norm'] = (df['composite_score'] / 100.0).fillna(0.5)

    results_list = []

    # ─────────────────────────────────────────────────────────────
    # Model 1: Attackers (FW, AM_WG)
    # ─────────────────────────────────────────────────────────────
    att_mask = (df['pos_group'].isin(['FW', 'AM_WG'])) & (df['90s'] >= 4.0)
    att_df = df[att_mask].copy()

    if len(att_df) >= 10:
        att_features = ['sh_per90', 'sot_per90', 'sot_pct', 'fld_per90', 'crs_per90', 'composite_norm', '90s']
        att_target = att_df['ga_per90']
        
        att_model = RandomForestRegressor(n_estimators=120, max_depth=5, random_state=42)
        att_model.fit(att_df[att_features], att_target)
        
        att_df['actual_target_metric'] = att_target.round(3)
        att_df['predicted_target_metric'] = att_model.predict(att_df[att_features]).round(3)
        att_df['target_metric_name'] = 'Goal Contributions (G+A/90)'
        att_df['performance_differential'] = (att_df['actual_target_metric'] - att_df['predicted_target_metric']).round(3)
        
        def eval_att(diff):
            if diff >= 0.15:
                return 'Clinical Overperformer'
            elif diff <= -0.15:
                return 'High Potential / Underconverting'
            return 'Performing As Expected'
        
        att_df['efficiency_status'] = att_df['performance_differential'].apply(eval_att)
        results_list.append(att_df)

    # ─────────────────────────────────────────────────────────────
    # Model 2: Midfielders & Fullbacks/Wingbacks (CM_DM, FB_WB)
    # Target: Two-Way Dynamic Impact (Dual Attack & Defensive Actions)
    # ─────────────────────────────────────────────────────────────
    mid_mask = (df['pos_group'].isin(['CM_DM', 'FB_WB'])) & (df['90s'] >= 4.0)
    mid_df = df[mid_mask].copy()

    if len(mid_df) >= 10:
        # Two-Way Impact: Rewards playmaking/creation, crossing, and defensive turnover recovery
        mid_target = (mid_df['ga_per90'] * 1.5) + (mid_df['tkl_int_per90']) + (mid_df['crs_per90'] * 0.4)
        mid_features = ['crs_per90', 'tkl_int_per90', 'sh_per90', 'fld_per90', 'composite_norm', '90s']
        
        mid_model = RandomForestRegressor(n_estimators=120, max_depth=5, random_state=42)
        mid_model.fit(mid_df[mid_features], mid_target)
        
        mid_df['actual_target_metric'] = mid_target.round(3)
        mid_df['predicted_target_metric'] = mid_model.predict(mid_df[mid_features]).round(3)
        mid_df['target_metric_name'] = 'Two-Way Dynamic Impact Index'
        mid_df['performance_differential'] = (mid_df['actual_target_metric'] - mid_df['predicted_target_metric']).round(3)
        
        def eval_mid(diff):
            if diff >= 0.40:
                return 'Elite Two-Way Engine'
            elif diff <= -0.40:
                return 'Underperforming Two-Way Impact'
            return 'Performing As Expected'
        
        mid_df['efficiency_status'] = mid_df['performance_differential'].apply(eval_mid)
        results_list.append(mid_df)

    # ─────────────────────────────────────────────────────────────
    # Model 3: Center-Backs (CB)
    # Target: Defensive Solidity Index (Tackles+Interceptions + Goal Suppression)
    # ─────────────────────────────────────────────────────────────
    cb_mask = (df['pos_group'] == 'CB') & (df['90s'] >= 4.0)
    cb_df = df[cb_mask].copy()

    if len(cb_df) >= 5:
        # Solidity Target: High tackles/interceptions + low goals conceded by team
        cb_target = cb_df['tkl_int_per90'] + ((2.5 - cb_df['team_ga90']).clip(lower=0) * 1.2)
        cb_features = ['tkl_int_per90', 'team_ga90', 'team_cs_pct', 'composite_norm', '90s']
        
        cb_model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
        cb_model.fit(cb_df[cb_features], cb_target)
        
        cb_df['actual_target_metric'] = cb_target.round(3)
        cb_df['predicted_target_metric'] = cb_model.predict(cb_df[cb_features]).round(3)
        cb_df['target_metric_name'] = 'Defensive Solidity Index'
        cb_df['performance_differential'] = (cb_df['actual_target_metric'] - cb_df['predicted_target_metric']).round(3)
        
        def eval_cb(diff):
            if diff >= 0.35:
                return 'Defensive Rock / High Solidity'
            elif diff <= -0.35:
                return 'Vulnerable / Concession Prone'
            return 'Performing As Expected'
        
        cb_df['efficiency_status'] = cb_df['performance_differential'].apply(eval_cb)
        results_list.append(cb_df)

    # ─────────────────────────────────────────────────────────────
    # Model 4: Goalkeepers (GK)
    # Target: Save Efficiency (Save%)
    # ─────────────────────────────────────────────────────────────
    processed_dir = os.path.join("data", "processed")
    gk_path = os.path.join(processed_dir, "goalkeeping.csv")
    if os.path.exists(gk_path):
        gk_raw = pd.read_csv(gk_path, encoding='utf-8-sig')
        gk_raw['MP'] = pd.to_numeric(gk_raw['MP'], errors='coerce').fillna(0)
        gk_qual = gk_raw[gk_raw['MP'] >= 6].copy()
        
        if len(gk_qual) >= 5:
            save_col = 'Save%' if 'Save%' in gk_qual.columns else 'save_pct'
            cs_col = 'CS%' if 'CS%' in gk_qual.columns else 'cs_pct'
            gk_qual['save_rate'] = pd.to_numeric(gk_qual[save_col], errors='coerce').fillna(65.0)
            gk_qual['cs_rate'] = pd.to_numeric(gk_qual[cs_col], errors='coerce').fillna(20.0)
            gk_qual['90s'] = pd.to_numeric(gk_qual['90s'], errors='coerce').fillna(6.0)
            gk_qual['SoTA'] = pd.to_numeric(gk_qual['SoTA'], errors='coerce').fillna(20.0)
            gk_qual['sota_per90'] = gk_qual['SoTA'] / gk_qual['90s'].replace(0, np.nan)
            
            # Merge composite score for GK
            gk_merged = pd.merge(gk_qual, df[['Player', 'composite_score']], on='Player', how='left')
            gk_merged['composite_norm'] = (gk_merged['composite_score'].fillna(50.0)) / 100.0
            
            gk_features = ['sota_per90', 'cs_rate', 'composite_norm', '90s']
            gk_target = gk_merged['save_rate']
            
            gk_model = RandomForestRegressor(n_estimators=80, max_depth=3, random_state=42)
            gk_model.fit(gk_merged[gk_features], gk_target)
            
            gk_merged['pos_group'] = 'GK'
            gk_merged['Gls'] = 0
            gk_merged['Ast'] = 0
            gk_merged['G+A'] = 0
            gk_merged['actual_ga_per90'] = 0.0
            gk_merged['actual_target_metric'] = gk_target.round(1)
            gk_merged['predicted_target_metric'] = gk_model.predict(gk_merged[gk_features]).round(1)
            gk_merged['target_metric_name'] = 'Save Efficiency (Save%)'
            gk_merged['performance_differential'] = (gk_merged['actual_target_metric'] - gk_merged['predicted_target_metric']).round(1)
            
            def eval_gk(diff):
                if diff >= 3.0:
                    return 'Elite Shot-Stopper'
                elif diff <= -3.0:
                    return 'Underperforming Shot-Stopper'
                return 'Performing As Expected'
            
            gk_merged['efficiency_status'] = gk_merged['performance_differential'].apply(eval_gk)
            results_list.append(gk_merged)

    # Combine all role predictions
    final_pred = pd.concat(results_list, ignore_index=True)
    
    # Backwards compatibility fields for frontend UI table
    final_pred['predicted_ga_per90'] = final_pred['predicted_target_metric']
    final_pred['actual_ga_per90'] = final_pred['actual_target_metric']
    final_pred['ga_differential'] = final_pred['performance_differential']

    save_cols = [
        'Player', 'Squad', 'Pos', 'pos_group', '90s', 'Gls', 'Ast', 'G+A',
        'target_metric_name', 'actual_target_metric', 'predicted_target_metric',
        'performance_differential', 'efficiency_status', 'composite_score',
        'predicted_ga_per90', 'actual_ga_per90', 'ga_differential'
    ]
    avail_cols = [c for c in save_cols if c in final_pred.columns]
    final_pred = final_pred[avail_cols].sort_values('composite_score', ascending=False).reset_index(drop=True)

    pred_path = os.path.join(metrics_dir, "player_ml_predictions.csv")
    final_pred.to_csv(pred_path, index=False, encoding='utf-8-sig')
    print(f"  [Holistic ML Player Models] Saved {len(final_pred)} role-tailored player predictions to {pred_path}")

    # Visualization
    plt.figure(figsize=(11, 6))
    palette = {
        'Clinical Overperformer': '#10B981',
        'Elite Two-Way Engine': '#059669',
        'Defensive Rock / High Solidity': '#34D399',
        'Elite Shot-Stopper': '#047857',
        'Performing As Expected': '#6366F1',
        'High Potential / Underconverting': '#F59E0B',
        'Underperforming Two-Way Impact': '#EF4444',
        'Vulnerable / Concession Prone': '#DC2626',
        'Underperforming Shot-Stopper': '#B91C1C'
    }
    sns.scatterplot(
        data=final_pred,
        x='predicted_target_metric',
        y='actual_target_metric',
        hue='efficiency_status',
        palette=palette,
        s=65,
        alpha=0.85
    )
    plt.title("Player Holistic Predicted vs Actual Impact by Tactical Role", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Metric Value (Model Expectation + Caliber Prior)", fontsize=11)
    plt.ylabel("Actual Metric Value", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ml_player_predictions.png"), dpi=300)
    plt.close()

    return final_pred

def train_match_prediction_model(processed_dir, metrics_dir, output_dir):
    """
    Train an ML Match Outcome Model to predict Home Win, Draw, or Away Win probabilities.
    Includes team rolling form, defensive stats, and home/away goal scoring rates.
    """
    matches_path = os.path.join(processed_dir, "matches.csv")
    matches_df = pd.read_csv(matches_path, encoding='utf-8-sig')

    # Filter completed matches
    matches_df = matches_df.dropna(subset=['Score']).copy()
    matches_df = matches_df[matches_df['Score'].str.contains(r'\d', na=False)].copy()

    parsed = matches_df['Score'].apply(parse_score)
    matches_df['home_gls'] = [p[0] for p in parsed]
    matches_df['away_gls'] = [p[1] for p in parsed]
    matches_df = matches_df.dropna(subset=['home_gls', 'away_gls']).copy()
    matches_df['Wk'] = pd.to_numeric(matches_df['Wk'], errors='coerce').fillna(0)
    matches_df = matches_df.sort_values(['Wk', 'Date']).reset_index(drop=True)

    # Determine target: 'H' (Home Win), 'D' (Draw), 'A' (Away Win)
    def get_result(row):
        if row['home_gls'] > row['away_gls']:
            return 'H'
        elif row['home_gls'] < row['away_gls']:
            return 'A'
        return 'D'

    matches_df['Result'] = matches_df.apply(get_result, axis=1)

    # Sequential tracker
    team_history = {}
    match_features = []

    for idx, row in matches_df.iterrows():
        home = row['Home']
        away = row['Away']

        h_hist = team_history.get(home, [])
        a_hist = team_history.get(away, [])

        h_pts_5 = sum([x['pts'] for x in h_hist[-5:]]) if h_hist else 0
        h_gf_5 = sum([x['gf'] for x in h_hist[-5:]]) if h_hist else 0
        h_ga_5 = sum([x['ga'] for x in h_hist[-5:]]) if h_hist else 0

        a_pts_5 = sum([x['pts'] for x in a_hist[-5:]]) if a_hist else 0
        a_gf_5 = sum([x['gf'] for x in a_hist[-5:]]) if a_hist else 0
        a_ga_5 = sum([x['ga'] for x in a_hist[-5:]]) if a_hist else 0

        feat = {
            'Wk': row['Wk'],
            'Home': home,
            'Away': away,
            'h_pts_5': h_pts_5,
            'h_gf_5': h_gf_5,
            'h_ga_5': h_ga_5,
            'a_pts_5': a_pts_5,
            'a_gf_5': a_gf_5,
            'a_ga_5': a_ga_5,
            'form_diff': h_pts_5 - a_pts_5,
            'gd_diff': (h_gf_5 - h_ga_5) - (a_gf_5 - a_ga_5),
            'Result': row['Result']
        }
        match_features.append(feat)

        h_g = int(row['home_gls'])
        a_g = int(row['away_gls'])
        h_pts = 3 if h_g > a_g else (1 if h_g == a_g else 0)
        a_pts = 3 if a_g > h_g else (1 if h_g == a_g else 0)

        if home not in team_history:
            team_history[home] = []
        if away not in team_history:
            team_history[away] = []

        team_history[home].append({'pts': h_pts, 'gf': h_g, 'ga': a_g})
        team_history[away].append({'pts': a_pts, 'gf': a_g, 'ga': h_g})

    feat_df = pd.DataFrame(match_features)

    X = feat_df[['h_pts_5', 'h_gf_5', 'h_ga_5', 'a_pts_5', 'a_gf_5', 'a_ga_5', 'form_diff', 'gd_diff']]
    y = feat_df['Result']

    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42)
    rf_clf.fit(X, y)

    cv_scores = cross_val_score(rf_clf, X, y, cv=5)
    print(f"  [ML Match Model] 5-Fold Cross Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    probs = rf_clf.predict_proba(X)
    classes = list(rf_clf.classes_)
    h_idx = classes.index('H')
    d_idx = classes.index('D')
    a_idx = classes.index('A')

    feat_df['prob_home_win'] = (probs[:, h_idx] * 100).round(1)
    feat_df['prob_draw'] = (probs[:, d_idx] * 100).round(1)
    feat_df['prob_away_win'] = (probs[:, a_idx] * 100).round(1)
    feat_df['predicted_result'] = rf_clf.predict(X)

    save_path = os.path.join(metrics_dir, "match_ml_predictions.csv")
    feat_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    cm = confusion_matrix(y, feat_df['predicted_result'], labels=['H', 'D', 'A'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Home Win', 'Draw', 'Away Win'], yticklabels=['Home Win', 'Draw', 'Away Win'])
    plt.title("Match Outcome Predictor Confusion Matrix", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Predicted Outcome")
    plt.ylabel("Actual Outcome")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ml_match_confusion_matrix.png"), dpi=300)
    plt.close()

    return feat_df, rf_clf, team_history

def predict_fixture(home_team, away_team, processed_dir, metrics_dir, model=None, team_history=None):
    """
    Forecast a match between any two Premier League clubs using their latest form.
    Returns win probabilities and favored outcome.
    """
    if model is None or team_history is None:
        _, model, team_history = train_match_prediction_model(processed_dir, metrics_dir, "outputs")

    h_hist = team_history.get(home_team, [])
    a_hist = team_history.get(away_team, [])

    h_pts_5 = sum([x['pts'] for x in h_hist[-5:]]) if h_hist else 6
    h_gf_5 = sum([x['gf'] for x in h_hist[-5:]]) if h_hist else 5
    h_ga_5 = sum([x['ga'] for x in h_hist[-5:]]) if h_hist else 5

    a_pts_5 = sum([x['pts'] for x in a_hist[-5:]]) if a_hist else 6
    a_gf_5 = sum([x['gf'] for x in a_hist[-5:]]) if a_hist else 5
    a_ga_5 = sum([x['ga'] for x in a_hist[-5:]]) if a_hist else 5

    feat = pd.DataFrame([{
        'h_pts_5': h_pts_5,
        'h_gf_5': h_gf_5,
        'h_ga_5': h_ga_5,
        'a_pts_5': a_pts_5,
        'a_gf_5': a_gf_5,
        'a_ga_5': a_ga_5,
        'form_diff': h_pts_5 - a_pts_5,
        'gd_diff': (h_gf_5 - h_ga_5) - (a_gf_5 - a_ga_5)
    }])

    probs = model.predict_proba(feat)[0]
    classes = list(model.classes_)
    h_p = float(round(probs[classes.index('H')] * 100, 1))
    d_p = float(round(probs[classes.index('D')] * 100, 1))
    a_p = float(round(probs[classes.index('A')] * 100, 1))

    predicted = model.predict(feat)[0]
    outcome_map = {'H': f'{home_team} Win', 'D': 'Draw', 'A': f'{away_team} Win'}

    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_prob': h_p,
        'draw_prob': d_p,
        'away_win_prob': a_p,
        'predicted_outcome': outcome_map[predicted]
    }

def run_ml_pipeline():
    print("Running Machine Learning Prediction Pipeline...")
    metrics_dir = os.path.join("data", "metrics")
    processed_dir = os.path.join("data", "processed")
    output_dir = "outputs"
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/2] Training Role-Specific Holistic Player Performance Predictor...")
    player_preds = train_player_prediction_model(metrics_dir, output_dir)
    print("  Saved: data/metrics/player_ml_predictions.csv")

    print("\n[2/2] Training Match Outcome & Win Probability Predictor...")
    match_preds, match_model, history = train_match_prediction_model(processed_dir, metrics_dir, output_dir)
    print("  Saved: data/metrics/match_ml_predictions.csv")

    sample = predict_fixture("Arsenal", "Chelsea", processed_dir, metrics_dir, match_model, history)
    print(f"\nSample Prediction - Arsenal vs Chelsea: {sample}")

if __name__ == "__main__":
    run_ml_pipeline()
