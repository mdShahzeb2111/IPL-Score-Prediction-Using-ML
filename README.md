# 🏏 IPL Score Prediction

A machine learning project that predicts the final score of an IPL (Indian Premier League) innings using real-time match data. This project can be used as a tool to estimate the target a team might set while batting first.

## 🔍 Project Overview

The goal of this project is to predict the final score of an IPL match's first innings using historical IPL data and live inputs such as:

- Current score
- Overs completed
- Wickets fallen
- Current run rate
- Batting team
- Bowling team

This can help fans, broadcasters, and analysts get insights into potential outcomes as the match progresses.

## 🧠 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn (for visualization)
- Jupyter Notebook
- Streamlit (for web app deployment, if applicable)
- Pickle/Joblib (for model serialization)

## 📂 Dataset

- Source: Kaggle or official IPL datasets
- Features include:
  - Batsman, Bowler, Venue
  - Batting Team & Bowling Team
  - Runs, Wickets, Overs
  - Match context (powerplay, death overs, etc.)

## ⚙️ How It Works

1. Data Cleaning & Preprocessing
2. Feature Engineering (e.g., encoding teams, overs breakdown)
3. Model Training using regression algorithms like:
   - Linear Regression
   - Random Forest Regressor
   - XGBoost
4. Model Evaluation using RMSE, R² Score
5. Predictions on live match inputs

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/mdShahzeb2111/IPL-Score-Prediction-Using-ML/edit/master.git
cd ipl-score-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook or Streamlit app
jupyter notebook  # or
streamlit run app.py
