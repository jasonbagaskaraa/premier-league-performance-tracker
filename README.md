# ⚽ Premier League Player Performance Tracker & Analytics Hub

A full-stack sports analytics and machine learning platform for tracking, evaluating, and predicting Premier League player and team performances using real 2025/2026 season data.

---

## 🌟 Key Features

- **End-to-End Data Pipeline**: Robust data ingestion, cleaning, normalization, and per-90 metrics derivation from FBref statistics.
- **Composite Scoring System**: Custom position-tailored performance rating algorithms (Attackers, Midfielders, Defenders, Goalkeepers).
- **Machine Learning Models**:
  - **Player Performance Regression**: Predicts future goals + assists contributions based on underlying metrics (xG, xAG, SCA, shots on target).
  - **Match Outcome Prediction**: Multi-class classifier predicting Home Win, Draw, or Away Win using rolling team form, home advantage, and attack/defense strengths.
- **Interactive Web Dashboard**: Modern responsive UI with live metric leaderboards, team rankings, form trends, and ML prediction simulator.
- **Exploratory Data Analysis**: Pre-built Jupyter notebooks and publication-ready Seaborn/Matplotlib visualizations in `outputs/`.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd "Premier League Performance Tracker"
pip install -r requirements.txt
```

### 2. Run the Complete Data & ML Pipeline

```bash
python main.py
```
This executes the automated workflow:
1. Data cleaning and schema validation (`src/cleaner.py`)
2. Per-90 and advanced metric derivation (`src/metrics.py`)
3. Composite score generation (`src/scoring.py`)
4. Team form and trend analysis (`src/trend.py`)
5. Chart generation to `outputs/` (`src/visualize.py`)
6. ML model training & evaluation (`src/predict.py`)

### 3. Launch Web Dashboard

```bash
python app.py
```
Open your browser and navigate to **`http://localhost:5000`**.

### 4. Run Unit Tests

```bash
python -m unittest discover tests
```

---

## 📁 Repository Structure

```
├── app.py                     # Flask API backend for web dashboard
├── main.py                    # Master pipeline runner
├── requirements.txt           # Project dependencies
├── .gitignore                 # Standard Python/IDE ignores
├── dashboard/                 # Frontend Web UI
│   ├── index.html             # Single-page dashboard application
│   ├── style.css              # Custom styling & dark-mode theme
│   └── app.js                 # Dynamic UI logic, charts, and API hooks
├── src/                       # Core Analytics & ML Modules
│   ├── cleaner.py             # Data sanitation & validation
│   ├── loader.py              # Data loader utilities
│   ├── metrics.py             # Derived football metrics (per-90, ratios)
│   ├── mock_data.py           # Fallback mock data generator
│   ├── predict.py             # Scikit-learn regression & match classifier
│   ├── scraper.py             # FBref data acquisition scraper
│   ├── scoring.py             # Weighted composite position scores
│   ├── trend.py               # Rolling form & team momentum calculation
│   └── visualize.py           # Matplotlib / Seaborn visualization pipeline
├── data/
│   ├── raw/                   # Raw FBref season datasets
│   ├── processed/             # Cleaned datasets ready for analytics
│   └── metrics/               # Derived metrics and composite rating CSVs
├── notebooks/
│   └── player_performance_evaluation.ipynb  # Interactive data analysis
├── outputs/                   # Exported charts & confusion matrices
└── tests/
    └── test_pipeline.py       # Automated unit test suite
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Data Engineering & Analytics**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Visualization**: Matplotlib, Seaborn, Chart.js
- **Backend / Web API**: Flask, Flask-CORS
- **Frontend**: HTML5, Vanilla CSS, JavaScript
