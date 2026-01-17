# Sales Performance Analysis & 90-Day Forecasting Dashboard

## 📌 Project Overview

This project focuses on analyzing historical retail sales data and building a time-series forecasting model to predict future revenue. It simulates a real-world business analytics and consulting scenario where data-driven insights support strategic decision-making such as inventory planning, revenue forecasting, and performance evaluation.

The project follows an end-to-end analytics workflow:
- Data generation and cleaning
- Exploratory data analysis (EDA)
- Time-series modeling
- Forecasting
- Executive-level visualization

---

## 🎯 Business Objective

- Understand sales trends and seasonality  
- Identify top-performing product categories  
- Forecast sales for the next 90 days  
- Support business planning and decision-making  

---

## 🔍 Key Features

- Analysis of **2 years of daily retail sales data** (730 days)
- Category-wise revenue and performance analysis across 5 product categories
- Time-series decomposition (trend, seasonality, residuals)
- Holt-Winters exponential smoothing forecasting model with **weekly seasonality**
- Model evaluation using MAE, RMSE, and MAPE metrics
- 90-day forward-looking sales forecast
- Interactive executive dashboard using Plotly

---

## 📊 Outputs

The project generates the following outputs:

| Output File | Description |
|------------|-------------|
| `category_revenue.png` | Revenue comparison across product categories |
| `daily_sales_trend.png` | Overall sales trend over 2-year period |
| `time_series_decomposition.png` | Trend, seasonal, and residual components |
| `forecast_validation.png` | Actual vs predicted sales comparison |
| `sales_forecast_90_days.csv` | Forecasted sales for next 90 days |
| `sales_dashboard.html` | Interactive sales forecasting dashboard |

---

## 🛠️ Tech Stack

**Programming Language:** Python 3.x

**Libraries:**
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Statistical Modeling:** Statsmodels
- **Machine Learning:** Scikit-learn

**Tools:** Git, GitHub

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Harshit-17-04/sales-forecasting-dashboard.git
cd sales-forecasting-dashboard
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the script
```bash
python sales_forecasting.py
```

### 4️⃣ View outputs
- Static visualizations will be saved as PNG files in the project directory
- Open `sales_dashboard.html` in your browser for the interactive dashboard
- Review `sales_forecast_90_days.csv` for detailed forecast data

---

## 📈 Business Impact

✅ **Revenue Planning:** Helps forecast future revenue for short-term planning and budgeting  
✅ **Operational Efficiency:** Supports inventory and staffing decisions based on predicted demand  
✅ **Strategic Insights:** Identifies high-performing product categories for focused marketing  
✅ **Decision Support:** Demonstrates how analytics can guide business strategy and resource allocation

---

## 🧠 Key Learnings

- Practical application of time-series forecasting in retail context
- Handling seasonality and trend components in sales data
- Model validation and performance evaluation techniques
- Translating technical analysis into business insights
- Building executive-friendly, interactive dashboards

---

## 📌 Future Enhancements

- [ ] Integrate real-world retail datasets (e.g., from Kaggle or public APIs)
- [ ] Add confidence intervals and prediction bounds to forecasts
- [ ] Implement automated data ingestion pipeline
- [ ] Compare multiple forecasting models (ARIMA, Prophet, LSTM)
- [ ] Deploy interactive dashboard using Streamlit or Flask
- [ ] Add anomaly detection for unusual sales patterns
- [ ] Incorporate external factors (holidays, promotions, economic indicators)

---

## 👤 Author

**Harshit Srivastava**  
Aspiring Data Analyst | Python | Data Analytics | Business Intelligence

📧 [harshitsri1704@gmail.com]  
💼 [www.linkedin.com/in/harshit-srivastava-14a41a30a]  


---

## 🔗 Repository Link

👉 **https://github.com/Harshit-17-04/sales-forecasting-dashboard**

---

## ✅ Why This Project Stands Out

✔ **Business-Focused:** Clear business objective aligned with consulting frameworks  
✔ **End-to-End Workflow:** Complete analytics pipeline from data to insights  
✔ **Professional Structure:** Organized like a real consulting engagement  
✔ **Technical Rigor:** Demonstrates proficiency in time-series forecasting and statistical modeling  
✔ **Communication Skills:** Translates complex analysis into actionable business insights  
✔ **Interview-Ready:** Showcases technical + business thinking valued at firms like Deloitte

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Last Updated: January 2026*
