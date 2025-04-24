# pollution-hotspot-detector
<h1 align="center">
  <a href="https://github.com/yourusername/aqi-hotspot-detector">
    Pollution Hotspot Detection and AQI Prediction
  </a>
</h1>

<div align="center">
  A machine learning-based software system to detect pollution hotspots and predict Air Quality Index (AQI) using environmental pollutant data.
</div>
<hr>

<details>
<summary>Table of Contents</summary>

- [Description](#description)  
- [Links](#links)  
- [Tech Stack](#tech-stack)  
- [Progress](#progress)  
- [Future Scope](#future-scope)  
- [Applications](#applications)  
- [Usage](#usage)  
- [Team Members](#team-members)  
 

</details>

---

## 📌 Description

This project focuses on **Pollution Hotspot Detection** and **AQI Prediction** using supervised machine learning techniques. We use the **Random Forest algorithm** to predict AQI from environmental pollutant levels such as PM2.5, PM10, NO₂, SO₂, CO, and O₃.

Using real-world pollution data, the system identifies high-risk areas (hotspots) and visualizes AQI trends over time. The project is entirely software-based and does **not require any hardware integration**.

---

## 🔗 Links

- [GitHub Repository](https://github.com/aryaborkar06/pollution-hotspot-detector)


---

## 🤖 Tech Stack
Language: Python
Platform: Google Colab
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
Tools: Excel, OpenPyXL, AQI.in (manual data)
Custom: CPCB AQI calculator (manual implementation)

## 📈 Progress

### ✅ Fully Implemented
- Data Cleaning and Preprocessing
- Pollutant Data Analysis  
- Random Forest Model Training  
- Daily AQI Prediction  
- Month-wise AQI Visualization  
- Hotspot Detection via AQI Thresholds  
- Line Plots and Bar Charts for Reporting  

## 🔮 Future Scope
Real-time AQI Prediction
→ Integration with live sensors or APIs for continuous monitoring.
Geospatial Heatmaps
→ Visualize hotspots on city maps using GPS data.
Mobile App Deployment
→ Develop an app for public alerts and AQI tracking.
Weather-Integrated Prediction
→ Improve accuracy using temperature, humidity, and wind data.
Social Media Alerts
→ Auto-post warnings for affected areas to Twitter or WhatsApp.
Edge/IoT Integration
→ Deploy lightweight models on local air quality sensors. 


## 💡 Applications
Health Alerts – Warn people in polluted areas.
City Planning – Avoid building in high-pollution zones.
Policy Making – Support air quality regulations.
Smart Cities – Real-time AQI monitoring.
Traffic Control – Reduce emissions in hotspots.

## 👨‍💻 Team Members
[Arya] (https://github.com/aryaborkar06)  
[Dhriti] (https://github.com/dhriti2208)

## 💻 Example Code

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('pollution_data.csv')
df.rename(columns={'PM 2.5': 'PM2.5', 'PM 10': 'PM10'}, inplace=True)
df['Year'] = 2021

# Preprocessing
df.replace('-', pd.NA, inplace=True)
for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)

# Features and Target
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Year']]
y = df['AQI']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Predict and visualize
df['Predicted_AQI'] = model.predict(X)
df['Date'] = pd.to_datetime(df['Date'])

plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['AQI'], label='Actual AQI', color='green')
plt.plot(df['Date'], df['Predicted_AQI'], label='Predicted AQI', color='blue', linestyle='--')
plt.title("Actual vs Predicted AQI (2021 – Daily Trend)")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
