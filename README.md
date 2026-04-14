# 📈 Financial News Sentiment Analysis for NIFTY 50 Prediction

A data-driven machine learning project that analyzes financial news sentiment to predict the **direction and magnitude of NIFTY 50 market movements**. This system integrates NLP (FinBERT), statistical validation, and regression models, and is deployed as an interactive web application on Hugging Face Spaces.

---

## 🚀 Live Deployment

🔗 https://huggingface.co/spaces/Sowjanya2408/bank-nifty-predictor

---

## 🧠 Project Overview

Financial markets react continuously to news, but quantifying this relationship is challenging. This project builds a complete pipeline to:

* Extract sentiment from financial news
* Validate its statistical relationship with market returns
* Predict market direction and return magnitude

The system is based on **4 years of financial news data (2022–2026)** covering all NIFTY 50 companies and demonstrates that **news sentiment carries measurable predictive power**.

---

## 📊 Key Results

* 📈 **Logistic Regression Accuracy:** 68.5%
* 📉 **Ridge Regression R² Score:** 0.235
* 📊 **Chi-Square Test:**

  * χ² = 124.36
  * p-value ≈ 9.92 × 10⁻28
  * Result: Statistically significant relationship

These results confirm that **news sentiment is not random and has predictive relevance**.

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **NLP Model:** FinBERT
* **Machine Learning Models:**

  * Logistic Regression (direction prediction)
  * Ridge Regression (return magnitude prediction)
* **Statistical Analysis:** Chi-Square Test
* **Framework:** Gradio
* **Deployment:** Hugging Face Spaces

---

## 📂 Dataset

* 📄 **Total Articles Collected:** 99,341
* 🧹 **After Deduplication:** 74,558
* 📊 **Daily Observations:** ~981
* 🏢 **Coverage:** All 50 NIFTY companies
* 📅 **Time Period:** April 2022 – April 2026

Sources:

* GDELT
* GNews
* Yahoo Finance (`^NSEI`)

---

## ⚙️ Methodology

### 1. Data Collection

* Financial news scraped for all NIFTY 50 companies
* Market data retrieved using `yfinance`

---

### 2. Data Preprocessing

* Removed duplicate articles (~25%)
* Cleaned text (URLs, symbols, whitespace)
* Combined title + body for analysis

---

### 3. Sentiment Analysis

* Used **FinBERT** to classify articles as:

  * Positive (+1)
  * Neutral (0)
  * Negative (−1)

* Long articles handled using **overlapping token segmentation**

---

### 4. Feature Engineering

* Daily sentiment aggregation:

  * Modal sentiment → classification feature
  * Mean sentiment score → regression feature

* Applied **weekend effect correction** to align news with trading days

---

### 5. Statistical Validation

* **Chi-Square Test of Independence**
* Confirmed non-random relationship between sentiment and returns

---

### 6. Modeling

#### 📈 Logistic Regression

* Predicts market direction
* Accuracy: **68.5%**

#### 📉 Ridge Regression

* Predicts return magnitude
* R²: **0.235**
* Helps reduce overfitting via regularization

---

## 🧩 Project Structure

```id="1n3w7l"
bank-nifty-predictor/
│
├── app.py              # Web app (Gradio interface)
├── model.py            # Model logic and prediction
├── requirements.txt    # Dependencies
├── Models              #Contains the trained models
├── Notebook            #Code for articles collection and sentiment analysis
├── Data                #Articles data
└── README.md           # Documentation
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/Sowjanya-Ashwath/Financial-News-Sentiment-Analysis-for-NIFTY-50-Prediction
pip install -r requirements.txt
python app.py
```

---

## 💡 Key Insights

* News sentiment shows **clear directional relationship** with market returns
* Positive sentiment → generally positive returns
* Negative sentiment → generally negative returns
* Sentiment explains **~23.5% of return variation** using a single feature

---

## ⚠️ Limitations

* Sentiment alone is not sufficient for full trading strategies
* Market behavior remains noisy and partially unpredictable
* Model performance varies across positive and negative cases

---

## 🔮 Future Scope

* Incorporate time-series models (LSTM, Transformers)
* Add macroeconomic indicators and technical features
* Real-time news ingestion pipeline
* Extend to sector-level or stock-level prediction

---

## 🎯 Applications

* Financial analytics
* Algorithmic trading research
* NLP in finance
* Market sentiment tracking

