# TCS Stock Price Analysis & Forecasting Using Machine Learning

##  Project Overview

This project analyzes and forecasts the daily closing price of **Tata Consultancy Services (TCS)** stock using statistical and machine learning models.

The objective was to build a reliable predictive model using historical market data, engineered financial indicators, and time-series techniques — while comparing traditional regression methods against advanced models like XGBoost and LSTM.

---

##  Problem Statement

Stock price behavior reflects both trend-following and short-term fluctuations.  
The goal of this project was to:

- Forecast daily **Closing Price**
- Engineer meaningful financial indicators
- Compare linear, ensemble, and deep learning models
- Identify the best trade-off between accuracy and complexity

---

##  Dataset

The dataset includes **4,000+ trading records (2002–2023)** containing:

- Open
- High
- Low
- Close
- Volume
- Dividends
- Stock Splits

Additional files:
- `TCS_stock_history.csv`
- `TCS_stock_action.csv`
- `TCS_stock_info.csv`

---

##  Exploratory Data Analysis (EDA)

Key observations:

- Strong long-term bullish trend (accelerated post-2020)
- OHLC values show near-perfect daily correlation
- Moving averages (30/50/200-day) reveal crossover points during crises (2008, 2020)
- Volume spikes pre-2016, stabilizes afterward

Correlation analysis showed:
- Open, High, Low, Prev_Close strongly correlated with Close
- Volume and Dividends had minimal impact

---

##  Feature Engineering

Over **30+ engineered features** were created, including:

###  Technical Indicators
- MA_5, MA_20, MA_30, MA_50, MA_200
- Rolling_Max_10, Rolling_Min_10
- Volatility_10

###  Lag Features
- Prev_Close
- Prev_Close_3

###  Price Relationships
- High_Low_Spread
- Open_Close_Ratio
- Price_Volume Interaction

###  Time Features
- Year
- Month
- Day
- Day_of_Week

---

##  Feature Selection

Multiple selection techniques were applied:

- Recursive Feature Elimination (RFE)
- Random Forest Feature Importance
- SelectKBest
- Correlation Analysis

Final selected features included:

- High
- Low
- Open
- Prev_Close
- Prev_Close_3
- MA_5
- MA_20
- MA_30
- MA_50
- MA_200
- Rolling_Max_10
- Rolling_Min_10
- Year
- Close_Month_Interaction
- High_Low_Spread
- Price_Volume

---

##  Models Implemented

### 1. Linear Regression (Baseline Model)

Features used (top via RFE):
- High
- Low
- Open
- Prev_Close
- MA_5

**Performance:**
- MSE: **42.27**
- R² Score: **0.99994**

---

### 2. XGBoost Regressor

**Performance:**
- MSE: **216.50**
- R² Score: **0.99969**

---

### 3. LSTM (30-Day Lookback Window)

Sequence-based deep learning model.

**Performance:**
- MAE: **34.49**
- MSE: **2162.93**

---

##  Results & Key Insight

 **Best Model: Linear Regression**

Despite testing advanced models like XGBoost and LSTM, the simple Linear Regression model significantly outperformed them.

This result highlights:

- Strong linear relationships in OHLC features
- High predictive power of engineered moving averages
- Importance of feature quality over model complexity

This demonstrates mature model selection — choosing performance and interpretability over unnecessary complexity.

---

---

##  Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib / Seaborn
