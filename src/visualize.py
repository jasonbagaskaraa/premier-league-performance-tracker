import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import os

# --- Style Config ---
DARK_BG = '#0d1117'
CARD_BG = '#161b22'
ACCENT_GREEN = '#3fb950'
ACCENT_BLUE = '#58a6ff'
ACCENT_PURPLE = '#bc8cff'
ACCENT_ORANGE = '#f0883e'
ACCENT_RED = '#f85149'
ACCENT_CYAN = '#39d2c0'
TEXT_COLOR = '#e6edf3'
GRID_COLOR = '#30363d'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': CARD_BG,
    'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': GRID_COLOR,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 11,
})


def plot_top_scorers(players_df, output_dir, top_n=15):
    df = players_df.copy()
    df['Gls'] = pd.to_numeric(df['Gls'], errors='coerce').fillna(0)
    df = df.sort_values('Gls', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(range(top_n), df['Gls'].values, color=ACCENT_GREEN, edgecolor='none', height=0.7)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"{row['Player']}  ({row['Squad']})" for _, row in df.iterrows()], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Goals', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Goal Scorers - Premier League 2025/26', fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.2)
    
    for bar, val in zip(bars, df['Gls'].values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                str(int(val)), va='center', fontsize=10, color=ACCENT_GREEN, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'top_scorers.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_shooting_efficiency(shooting_df, output_dir, min_shots=30):
    df = shooting_df.copy()
    for col in ['Sh', 'shot_accuracy', 'goal_conversion', 'Gls']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df = df[df['Sh'] >= min_shots]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df['shot_accuracy'], df['goal_conversion'],
                         s=df['Gls'] * 12 + 30, c=df['Gls'], cmap='YlGn',
                         edgecolors=ACCENT_GREEN, linewidth=0.5, alpha=0.85)
    
    top_scorers = df.nlargest(8, 'goal_conversion')
    for _, row in top_scorers.iterrows():
        ax.annotate(row['Player'], (row['shot_accuracy'], row['goal_conversion']),
                    textcoords="offset points", xytext=(8, 5), fontsize=8,
                    color=TEXT_COLOR, alpha=0.9)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Goals Scored', color=TEXT_COLOR, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)
    
    ax.set_xlabel('Shot Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Goal Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Shooting Efficiency (min {min_shots} shots) - PL 2025/26',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'shooting_efficiency.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_discipline_ranking(discipline_df, output_dir, top_n=15):
    df = discipline_df.copy()
    df['discipline_points'] = pd.to_numeric(df['discipline_points'], errors='coerce').fillna(0)
    df = df.sort_values('discipline_points', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = [ACCENT_RED if dp >= 10 else ACCENT_ORANGE if dp >= 5 else ACCENT_CYAN 
              for dp in df['discipline_points'].values]
    bars = ax.barh(range(top_n), df['discipline_points'].values, color=colors, height=0.7)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"{row['Player']}  ({row['Squad']})" for _, row in df.iterrows()], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Discipline Points (Yellow=1, Red=3)', fontsize=11, fontweight='bold')
    ax.set_title('Most Carded Players - Premier League 2025/26', fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.2)
    
    for bar, val in zip(bars, df['discipline_points'].values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                str(int(val)), va='center', fontsize=10, fontweight='bold', color=TEXT_COLOR)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'discipline_ranking.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_gk_performance(gk_df, output_dir, min_matches=10):
    df = gk_df.copy()
    df['MP'] = pd.to_numeric(df['MP'], errors='coerce').fillna(0)
    df['save_pct'] = pd.to_numeric(df['save_pct'], errors='coerce').fillna(0)
    df['cs_pct'] = pd.to_numeric(df['cs_pct'], errors='coerce').fillna(0)
    df = df[df['MP'] >= min_matches].sort_values('save_pct', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = np.arange(len(df))
    bar_height = 0.35
    
    bars1 = ax.barh(y_pos - bar_height/2, df['save_pct'].values, bar_height,
                    label='Save %', color=ACCENT_BLUE, edgecolor='none')
    bars2 = ax.barh(y_pos + bar_height/2, df['cs_pct'].values, bar_height,
                    label='Clean Sheet %', color=ACCENT_PURPLE, edgecolor='none')
    
    ax.set_yticks(y_pos)
    labels = [f"{row['Player']}  ({row['Squad']})" for _, row in df.iterrows()]
    safe_labels = [l.encode('ascii', 'replace').decode('ascii') for l in labels]
    ax.set_yticklabels(safe_labels, fontsize=10)
    ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Goalkeeper Performance (min {min_matches} matches) - PL 2025/26',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, facecolor=CARD_BG, edgecolor=GRID_COLOR)
    ax.grid(axis='x', alpha=0.15)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'gk_performance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_team_goals(teams_df, output_dir):
    df = teams_df.copy()
    df['Gls'] = pd.to_numeric(df['Gls'], errors='coerce').fillna(0)
    df = df.sort_values('Gls', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.YlGn(np.linspace(0.3, 0.95, len(df)))
    bars = ax.barh(range(len(df)), df['Gls'].values, color=colors, height=0.7)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Squad'].values, fontsize=11)
    ax.set_xlabel('Goals Scored', fontsize=12, fontweight='bold')
    ax.set_title('Goals Scored by Team - Premier League 2025/26', fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.2)
    
    for bar, val in zip(bars, df['Gls'].values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(int(val)), va='center', fontsize=10, fontweight='bold', color=TEXT_COLOR)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'team_goals.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_composite_rankings(composite_df, output_dir, top_n=15):
    df = composite_df.copy()
    df['composite_score'] = pd.to_numeric(df['composite_score'], errors='coerce').fillna(0)
    df = df.sort_values('composite_score', ascending=False).head(top_n)
    df = df.sort_values('composite_score', ascending=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(top_n), df['composite_score'].values, color=ACCENT_CYAN, edgecolor='none', height=0.65)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"{row['Player']} [{row['Pos']}] ({row['Squad']})" for _, row in df.iterrows()], fontsize=10)
    ax.set_xlabel('Composite Score Index (0-100)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Overall Players by Position-Aware Composite Index - Premier League 2025/26', fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.2)
    
    for bar, val in zip(bars, df['composite_score'].values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va='center', fontsize=10, fontweight='bold', color=TEXT_COLOR)

    plt.tight_layout()
    path = os.path.join(output_dir, 'composite_rankings.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_position_rankings(composite_df, output_dir, top_n=5):
    positions = [
        ('FW', 'Centre-Forwards', ACCENT_GREEN), 
        ('AM_WG', 'Attacking Midfielders & Wingers', ACCENT_BLUE), 
        ('CM_DM', 'Central / Defensive Midfielders', ACCENT_CYAN), 
        ('DF', 'Defenders', ACCENT_ORANGE),
        ('GK', 'Goalkeepers', ACCENT_PURPLE)
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (pos_code, pos_name, color) in enumerate(positions):
        ax = axes[idx]
        pos_col = 'pos_group' if 'pos_group' in composite_df.columns else 'Pos'
        pos_df = composite_df[composite_df[pos_col].astype(str).str.contains(pos_code, na=False)].copy()
        pos_df['composite_score'] = pd.to_numeric(pos_df['composite_score'], errors='coerce').fillna(0)
        pos_df = pos_df.sort_values('composite_score', ascending=False).head(top_n)
        pos_df = pos_df.sort_values('composite_score', ascending=True)

        bars = ax.barh(range(len(pos_df)), pos_df['composite_score'].values, color=color, height=0.6)
        ax.set_yticks(range(len(pos_df)))
        ax.set_yticklabels([f"{row['Player']} ({row['Squad']})" for _, row in pos_df.iterrows()], fontsize=10)
        ax.set_title(f'Top {top_n} {pos_name}', fontsize=12, fontweight='bold', color=TEXT_COLOR)
        ax.set_xlabel('Composite Score (0-100)', fontsize=10)
        ax.grid(axis='x', alpha=0.2)

        for bar, val in zip(bars, pos_df['composite_score'].values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va='center', fontsize=9, fontweight='bold', color=TEXT_COLOR)

    # Hide 6th unused subplot
    fig.delaxes(axes[5])

    plt.suptitle('Top Players by Position (Position-Aware Percentile Scoring Model)', fontsize=16, fontweight='bold', color=TEXT_COLOR)
    plt.tight_layout()
    path = os.path.join(output_dir, 'position_rankings.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_team_form_trend(form_df, output_dir):
    df = form_df.copy()
    df['pts_last_5'] = pd.to_numeric(df['pts_last_5'], errors='coerce').fillna(0)
    df = df.sort_values('pts_last_5', ascending=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = [ACCENT_GREEN if status == 'Hot Form' else ACCENT_BLUE if status == 'Stable Form' else ACCENT_RED
              for status in df['form_status'].values]
              
    bars = ax.barh(range(len(df)), df['pts_last_5'].values, color=colors, height=0.65)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Squad'].values, fontsize=10)
    ax.set_xlabel('Points in Last 5 Matches (Max 15)', fontsize=12, fontweight='bold')
    ax.set_title('Premier League Team Form Ranking (Last 5 Matches)', fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.2)
    
    for bar, val, status in zip(bars, df['pts_last_5'].values, df['form_status'].values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{int(val)} pts ({status})", va='center', fontsize=9, fontweight='bold', color=TEXT_COLOR)

    plt.tight_layout()
    path = os.path.join(output_dir, 'team_form_trend.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


if __name__ == "__main__":
    metrics_dir = os.path.join("data", "metrics")
    processed_dir = os.path.join("data", "processed")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    shooting_df = pd.read_csv(os.path.join(metrics_dir, "shooting_metrics.csv"))
    players_df = pd.read_csv(os.path.join(metrics_dir, "player_metrics.csv"))
    discipline_df = pd.read_csv(os.path.join(metrics_dir, "discipline_metrics.csv"))
    gk_df = pd.read_csv(os.path.join(metrics_dir, "gk_metrics.csv"))
    teams_df = pd.read_csv(os.path.join(processed_dir, "teams.csv"))

    plot_top_scorers(players_df, output_dir)
    plot_shooting_efficiency(shooting_df, output_dir)
    plot_discipline_ranking(discipline_df, output_dir)
    plot_gk_performance(gk_df, output_dir)
    plot_team_goals(teams_df, output_dir)
    
    print("\nAll visualizations saved to outputs/")
