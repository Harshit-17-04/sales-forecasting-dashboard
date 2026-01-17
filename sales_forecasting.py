# =========================================================
# Sales Performance Analysis & 90-Day Forecasting
# =========================================================

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from datetime import timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

print("[INFO] Script started")

# ---------------------------------------------------------
# 1. Generate Synthetic Retail Sales Data
# ---------------------------------------------------------

np.random.seed(42)

dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
trend = np.linspace(50000, 80000, len(dates))
seasonality = 15000 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
noise = np.random.normal(0, 5000, len(dates))

sales = trend + seasonality + noise

categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
records = []

for date, total_sale in zip(dates, sales):
    for cat in categories:
        share = np.random.uniform(0.1, 0.3)
        cat_sales = total_sale * share
        records.append({
            "Date": date,
            "Category": cat,
            "Sales": cat_sales,
            "Units_Sold": int(cat_sales / np.random.uniform(50, 200))
        })

df = pd.DataFrame(records)

print("[INFO] Dataset created")

# ---------------------------------------------------------
# 2. Category Performance Analysis
# ---------------------------------------------------------

category_summary = df.groupby("Category").agg({
    "Sales": ["sum", "mean"],
    "Units_Sold": "sum"
}).round(2)

category_summary.columns = ["Total_Sales", "Avg_Daily_Sales", "Total_Units"]
category_summary = category_summary.sort_values("Total_Sales", ascending=False)

print("[INFO] Category summary calculated")

# ---------------------------------------------------------
# 3. Category Revenue Plot
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
sns.barplot(x=category_summary.index, y=category_summary["Total_Sales"])
plt.title("Total Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("category_revenue.png")
plt.show(block=False)
plt.close()

print("[INFO] category_revenue.png saved")

# ---------------------------------------------------------
# 4. Daily Sales Trend
# ---------------------------------------------------------

daily_sales = df.groupby("Date")["Sales"].sum().reset_index()
ts = daily_sales.set_index("Date")["Sales"]

plt.figure(figsize=(14, 6))
plt.plot(daily_sales["Date"], daily_sales["Sales"])
plt.title("Daily Sales Trend (2022–2023)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("daily_sales_trend.png")
plt.show(block=False)
plt.close()

print("[INFO] daily_sales_trend.png saved")

# ---------------------------------------------------------
# 5. Time Series Decomposition
# ---------------------------------------------------------

decomposition = seasonal_decompose(ts, model="additive", period=365)
decomposition.plot()
plt.tight_layout()
plt.savefig("time_series_decomposition.png")
plt.show(block=False)
plt.close()

print("[INFO] time_series_decomposition.png saved")

# ---------------------------------------------------------
# 6. Train-Test Split
# ---------------------------------------------------------

train_size = int(len(ts) * 0.8)
train = ts[:train_size]
test = ts[train_size:]

print("[INFO] Train-test split completed")

# ---------------------------------------------------------
# 7. Holt-Winters Forecasting (Weekly Seasonality)
# ---------------------------------------------------------

model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=7
)

fit_model = model.fit()
predictions = fit_model.forecast(len(test))

mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))
mape = np.mean(np.abs((test - predictions) / test)) * 100

print("[INFO] Model trained")
print(f"[INFO] MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

# ---------------------------------------------------------
# 8. Forecast Validation Plot
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Actual Sales")
plt.plot(test.index, predictions, "--", label="Predicted Sales")
plt.legend()
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_validation.png")
plt.show(block=False)
plt.close()

print("[INFO] forecast_validation.png saved")

# ---------------------------------------------------------
# 9. Final Model & 90-Day Forecast
# ---------------------------------------------------------
# ---------------------------------------------------------
# 9. Final Model & 90-Day Forecast
# ---------------------------------------------------------

final_model = ExponentialSmoothing(
    ts,
    trend="add",
    seasonal="add",
    seasonal_periods=7
).fit()

future_forecast = final_model.forecast(90)
future_dates = pd.date_range(ts.index[-1] + timedelta(days=1), periods=90)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Sales": future_forecast
})

# Add error handling for file saving
try:
    forecast_df.to_csv("sales_forecast_90_days.csv", index=False)
    print("[INFO] sales_forecast_90_days.csv saved")
except PermissionError:
    # Try with a different filename
    alt_filename = f"sales_forecast_90_days_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    forecast_df.to_csv(alt_filename, index=False)
    print(f"[WARNING] Original file was locked. Saved as {alt_filename}")

# ---------------------------------------------------------
# 10. Interactive Dashboard
# ---------------------------------------------------------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ts.index[-180:],
    y=ts.values[-180:],
    mode="lines",
    name="Historical Sales"
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_forecast,
    mode="lines",
    name="90-Day Forecast",
    line=dict(dash="dash")
))

fig.update_layout(
    title="Sales Performance & 90-Day Forecast Dashboard",
    xaxis_title="Date",
    yaxis_title="Sales",
    template="plotly_white"
)

fig.write_html("sales_dashboard.html")

print("[INFO] sales_dashboard.html saved")

# ---------------------------------------------------------
# 11. Final Message
# ---------------------------------------------------------

print("[DONE] Script completed successfully")
