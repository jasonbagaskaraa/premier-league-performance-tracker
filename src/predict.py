import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix

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
    Train an ML Regression model (Random Forest) to predict expected Goal Contributions per 90 (xG+A/90)
    based on underlying attacking and performance metrics.
    Identifies clinical overperformers and high-potential underperformers.
    """
    comp_path = os.path.join(metrics_dir, "composite_scores.csv")
    if not os.path.exists(comp_path):
        raise FileNotFoundError(f"Missing {comp_path}")

    df = pd.read_csv(comp_path)

    # Filter for outfield players with at least 4.0 nineties (360+ minutes)
    outfield_df = df[(df['pos_group'] != 'GK') & (df['90s'] >= 4.0)].copy()

    # Derived rate features
    outfield_df['sh_per90'] = outfield_df['Sh'] / outfield_df['90s']
    outfield_df['sot_per90'] = outfield_df['SoT'] / outfield_df['90s']
    outfield_df['sot_pct'] = np.where(outfield_df['Sh'] > 0, outfield_df['SoT'] / outfield_df['Sh'], 0.0)
    outfield_df['fld_per90'] = outfield_df['Fld'] / outfield_df['90s']
    
    # Target: actual Goal Contributions per 90 (G+A per 90)
    outfield_df['target_ga_per90'] = (outfield_df['Gls'] + outfield_df['Ast']) / outfield_df['90s']

    feature_cols = ['sh_per90', 'sot_per90', 'sot_pct', 'fld_per90', 'tkl_int_per90', '90s']
    
    # One-hot encode pos_group
    pos_dummies = pd.get_dummies(outfield_df['pos_group'], prefix='pos', drop_first=True)
    X = pd.concat([outfield_df[feature_cols], pos_dummies], axis=1)
    y = outfield_df['target_ga_per90']

    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X, y)

    # Predictions & Residuals
    outfield_df['predicted_ga_per90'] = model.predict(X).round(3)
    outfield_df['actual_ga_per90'] = outfield_df['target_ga_per90'].round(3)
    outfield_df['ga_differential'] = (outfield_df['actual_ga_per90'] - outfield_df['predicted_ga_per90']).round(3)

    # Categorize clinical efficiency
    def classify_efficiency(diff):
        if diff >= 0.15:
            return 'Clinical Overperformer'
        elif diff <= -0.15:
            return 'High Potential / Underconverting'
        else:
            return 'Performing As Expected'

    outfield_df['efficiency_status'] = outfield_df['ga_differential'].apply(classify_efficiency)

    # Metrics
    mae = mean_absolute_error(y, outfield_df['predicted_ga_per90'])
    rmse = np.sqrt(mean_squared_error(y, outfield_df['predicted_ga_per90']))
    r2 = r2_score(y, outfield_df['predicted_ga_per90'])

    print(f"  [ML Player Model] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R² Score: {r2:.3f}")

    # Save predictions
    save_cols = ['Player', 'Squad', 'Pos', 'pos_group', '90s', 'Gls', 'Ast', 'G+A', 
                 'actual_ga_per90', 'predicted_ga_per90', 'ga_differential', 'efficiency_status', 'composite_score']
    pred_df = outfield_df[save_cols].sort_values('predicted_ga_per90', ascending=False).reset_index(drop=True)
    pred_path = os.path.join(metrics_dir, "player_ml_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')

    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=outfield_df,
        x='predicted_ga_per90',
        y='actual_ga_per90',
        hue='efficiency_status',
        palette={'Clinical Overperformer': '#10B981', 'Performing As Expected': '#6366F1', 'High Potential / Underconverting': '#F59E0B'},
        s=70,
        alpha=0.85
    )
    # Line of equality
    max_val = max(outfield_df['predicted_ga_per90'].max(), outfield_df['actual_ga_per90'].max()) + 0.1
    plt.plot([0, max_val], [0, max_val], '--', color='#9CA3AF', linewidth=1.5, label='Par Expected Rate')
    plt.title("Player Expected vs Actual Goal Contributions per 90 (ML Model)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted G+A per 90 (Model Expectation)", fontsize=11)
    plt.ylabel("Actual G+A per 90", fontsize=11)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ml_player_predictions.png"), dpi=300)
    plt.close()

    # Feature Importance Plot
    feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    feature_imp.plot(kind='barh', color='#3B82F6')
    plt.title("Feature Importance - Player Goal Contribution Predictor", fontsize=12, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ml_feature_importance.png"), dpi=300)
    plt.close()

    return pred_df, model

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

    # Build sequential team match tracker to get rolling stats without data leakage
    team_history = {}
    match_features = []

    for idx, row in matches_df.iterrows():
        home = row['Home']
        away = row['Away']

        # Get current pre-match rolling form for home team
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

        # Update history
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

    # Train/Test Split (Chronological: past matches train, later matches test)
    X = feat_df[['h_pts_5', 'h_gf_5', 'h_ga_5', 'a_pts_5', 'a_gf_5', 'a_ga_5', 'form_diff', 'gd_diff']]
    y = feat_df['Result']

    # Train Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42)
    rf_clf.fit(X, y)

    # Cross validation score
    cv_scores = cross_val_score(rf_clf, X, y, cv=5)
    print(f"  [ML Match Model] 5-Fold Cross Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Generate probabilities for matches
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

    # Confusion matrix visualization
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

    print("\n[1/2] Training Player Goal Contributions & Efficiency Predictor...")
    player_preds, player_model = train_player_prediction_model(metrics_dir, output_dir)
    print("  Saved: data/metrics/player_ml_predictions.csv")

    print("\n[2/2] Training Match Outcome & Win Probability Predictor...")
    match_preds, match_model, history = train_match_prediction_model(processed_dir, metrics_dir, output_dir)
    print("  Saved: data/metrics/match_ml_predictions.csv")

    # Sample fixture test
    sample = predict_fixture("Arsenal", "Chelsea", processed_dir, metrics_dir, match_model, history)
    print(f"\nSample Prediction - Arsenal vs Chelsea: {sample}")

if __name__ == "__main__":
    run_ml_pipeline()
