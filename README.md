# Evergreen Stock Prediction & Sentiment Analysis

> A daily stock prediction platform that merges historical market data, sentiment analysis, and ensemble ML models, leveraging robust cloud infrastructure (Modal for daily scheduling, AWS S3 for logs, and Streamlit for UI – optionally hosted on Streamlit Community Cloud).

---

## 🚀 Project Overview

- **Stock Price Prediction & Sentiment Analysis**  
  Combines **historical market data**, **sentiment signals**, and **ensemble ML** models to forecast near-future prices and price direction.

- **Automated Cloud Orchestration**  
  - **Modal** handles daily scheduling (fetch data, run inference, upload logs).  
  - **AWS S3** stores daily outputs (CSV logs) for persistent storage.

- **Streamlit Frontend**  
  - An interactive UI enabling **real-time** data visualization and user exploration.
  - Deployable on **Streamlit Community Cloud** or self-hosted, as desired.

---

## 📂 Data Collection & Preparation

- **Data Sources**  
  - **Polygon** for historical market data and news data.  
  

- **Data Preprocessing**  
  - Handling missing values, detecting outliers, and aligning time-series data.
  - Feature engineering: moving averages (MA), RSI, MACD, etc.
  - **Scaling:** Using a single `scaler_x` to ensure consistent transformations across training, validation, and test sets.

---

## 🧠 Model Development & Training

- **Ensemble ML Approach**  
  - Evaluated multiple algorithms (XGBoost, Random Forest, Neural Networks).  
  - Employed ensemble blending to leverage each model’s strengths.

- **Inverse Scaling & Softmax**  
  - Integrated inverse scaling to transform model outputs back to real-world values.
  - Utilized a softmax-based approach to generate directional probabilities (up vs. down), allowing the model to function as both a price predictor and a direction predictor.

---

## 🚧 Top 5 Challenges & Solutions

1. **Scaling & Data Leakage**  
   - **Issue:** Multiple scalers for different subsets led to mismatched predictions.  
   - **Solution:** Unified the scaling approach by using a single fitted scaler across the entire pipeline.

2. **Modal Integration & Older Versions**  
   - **Issue:** Modal version mismatches (e.g., `0.73.x` vs. newer) led to environment conflicts and `asgi_app` errors.  
   - **Solution:** Updated the code to use the new Modal API (`@stub.function`, `@modal.asgi_app`) and pinned Python dependencies to maintain environment consistency.

3. **Sentiment Data Alignment**  
   - **Issue:** Time lags between stock data and news data resulted in merging inaccuracies.  
   - **Solution:** Aligned timestamps carefully, ensuring that sentiment data was merged only with the corresponding trading day.

4. **Daily Logs & AWS S3 Integration**  
   - **Issue:** The automated pipeline required a persistent storage solution for daily CSV logs.  
   - **Solution:** Configured the pipeline to push CSV logs to an AWS S3 bucket after each run for reliable storage and easy retrieval by Streamlit.

5. **Streamlit Hosting & Environment Mismatch**  
   - **Issue:** Discrepancies between the local development environment and Streamlit’s ephemeral cloud environment (e.g., differences in Python versions or missing packages).  
   - **Solution:** Standardized the environment with a pinned `requirements.txt` and, if necessary, used a `packages.txt` for system-level dependencies on Streamlit Community Cloud.

---

## ⚠️ Assumptions & Justifications

- **Historical Data Projects Future**  
  - Assumes past trends and technical signals remain relevant in the short term.
  
- **Sentiment Reflects Market Moves**  
  - Assumes that news sentiment significantly influences short-term price movements.
  
- **Stationary Features**  
  - Assumes that indicators like RSI and MACD exhibit near-stationary or cyclical behavior over short time windows.

---

## ☁️ Cloud Workflow

1. **Modal**  
   - Orchestrates a daily job (or cron task) to fetch data, run predictions, and save logs.
   - The pipeline is managed via `modal_backend.py` or an equivalent script.

2. **AWS S3**  
   - Receives CSV outputs from daily runs (e.g., `predictions_YYYYMMDD.csv`).
   - These logs are accessed by the Streamlit app for visualization.

3. **Streamlit**  
   - Displays real-time or daily aggregated data.
   - Can be deployed on Streamlit Community Cloud with a properly configured environment (`requirements.txt`, `packages.txt`).

---

## 📚 Deploying on Streamlit Community Cloud

1. **GitHub Repository Setup**  
   - Create a GitHub repository for your project.

2. **Connecting to Streamlit Community Cloud**  
   - Connect your GitHub repository in Streamlit Community Cloud.
   - Specify **`src/frontend/app.py`** as the main entry point.
   - Provide a `requirements.txt` with pinned dependencies.

3. **AWS S3 Integration**  
   - Store AWS credentials securely as secrets in the Streamlit app settings.
   - Ensure AWS credentials are not committed in plain text.

4. **Troubleshooting Environment Mismatches**  
   - Use `python --version` to verify the environment.
   - If needed, include a `packages.txt` for additional system dependencies.

---

## 🔮 Future Steps & Enhancements
   
1. **Expanded Ticker Coverage**  
   - Increase coverage from a few tickers to a broader watchlist with sector diversity.

2. **Alert & Notification Systems**  
   - Integrate Slack or email notifications for significant model updates or sentiment shifts.

3. **Advanced ML Techniques**  
   - Explore Transformers, or other advanced architectures for improved time-series predictions.
     
4. **Study Politicians Trades**
   - Use trade data of politicians to learn how to find the best ones, or just outright copy their trades.
---

## 🧠 Key Learnings & Insights

- **Consistent Data Preprocessing is Crucial**  
  - Using a single scaler and robust feature engineering enhances model reliability.
  
- **Cloud Deployment Requires Rigor**  
  - Proactively managing environment mismatches and API versioning issues can save debugging time.
  
- **Ensemble ML & Softmax Integration**  
  - Combining different models with softmax outputs effectively bridges regression with probabilistic directional forecasting.
  
- **Automation Strengthens Reliability**  
  - A pipeline that integrates Modal for scheduling, AWS S3 for storage, and Streamlit for visualization is both robust and scalable.
  
- **Iterative Problem Solving**  
  - Overcoming challenges such as data alignment and environment issues reinforces the value of continuous learning and adaptation.

---

