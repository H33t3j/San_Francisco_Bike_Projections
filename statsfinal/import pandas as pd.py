import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. LOAD AND PREPROCESS
df = pd.read_csv('hour.csv')
df['dteday'] = pd.to_datetime(df['dteday'], dayfirst=True)
df['datetime'] = df['dteday'] + pd.to_timedelta(df['hr'], unit='h')
df = df.set_index('datetime').sort_index()

# 2. DENOISING: Remove white noise using a 3-hour centered rolling mean
# This smooths random jitter while preserving the 'rush hour' signal.
df['cnt_smooth'] = df['cnt'].rolling(window=3, center=True).mean().fillna(method='ffill').fillna(method='bfill')

# 3. FEATURE ENGINEERING
def create_features(data):
    d = data.copy()
    # Lag Features: $t-1$, $t-24$ (yesterday), $t-168$ (last week)
    d['lag_1h'] = d['cnt'].shift(1)
    d['lag_24h'] = d['cnt'].shift(24)
    d['lag_168h'] = d['cnt'].shift(168)
    
    # Rolling trend
    d['rolling_mean_24h'] = d['cnt'].shift(1).rolling(window=24).mean()
    
    # Rush Hour Logic: Flagging the 8 AM and 5 PM peaks on workdays
    d['is_rush_hour'] = d.apply(
        lambda x: 1 if (x['workingday'] == 1 and (7 <= x['hr'] <= 9 or 16 <= x['hr'] <= 18)) else 0, 
        axis=1
    )
    return d.dropna()

df_feat = create_features(df)

# 4. DATA SPLITTING (Train 2011 / Test 2012)
train = df_feat[df_feat['yr'] == 0]
test = df_feat[df_feat['yr'] == 1]

features = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 
            'temp', 'atemp', 'hum', 'windspeed', 'lag_1h', 'lag_24h', 'lag_168h', 
            'rolling_mean_24h', 'is_rush_hour']

# Log transformation: $\log(1+x)$ stabilizes the variance across seasons
X_train, y_train_log = train[features], np.log1p(train['cnt_smooth'])
X_test, y_test_actual = test[features], test['cnt']

# 5. OPTIMIZED GRADIENT BOOSTING MODEL
model = HistGradientBoostingRegressor(
    max_iter=300, 
    max_depth=10, 
    learning_rate=0.05, 
    random_state=42
)
model.fit(X_train, y_train_log)

# Predict and inverse log transform: $e^y - 1$
predictions = np.expm1(model.predict(X_test))

# 6. RESULTS AND METRICS
mae = mean_absolute_error(y_test_actual, predictions)
r2 = r2_score(y_test_actual, predictions)
print(f"Full Year Performance -> MAE: {mae:.2f}, R2: {r2:.2f}")

# 7. COMPARISON CHARTS
# Create a results DataFrame for plotting
results_df = pd.DataFrame({
    'Actual': y_test_actual.values,
    'Predicted': predictions
}, index=test.index)

# GRAPH 1: DAILY AGGREGATION (Full Year 2012)
# We aggregate to 'D' (daily) to see the long-term trend clearly
daily_results = results_df.resample('D').sum()

plt.figure(figsize=(15, 7))
plt.plot(daily_results.index, daily_results['Actual'], label='Actual (Daily)', color='black', alpha=0.3)
plt.plot(daily_results.index, daily_results['Predicted'], label='GB Prediction', color='red', linewidth=1.5)
plt.title('2012 Full Year: Actual vs Predicted Daily Bike Rentals')
plt.xlabel('Date')
plt.ylabel('Total Rentals')
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig('full_year_daily_chart.png')

# GRAPH 2: HOURLY SNIPPET (First Week of July)
# Zooming in to see the detail of the 'rush hour' peaks
july_snippet = results_df['2012-07-01':'2012-07-07']

plt.figure(figsize=(15, 7))
plt.plot(july_snippet.index, july_snippet['Actual'], label='Actual (Hourly)', color='gray', alpha=0.5, marker='o')
plt.plot(july_snippet.index, july_snippet['Predicted'], label='GB Prediction', color='red', linestyle='--', marker='x')
plt.title('Hourly Zoom-In: One Week Comparison (July 2012)')
plt.xlabel('Time')
plt.ylabel('Rental Count')
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig('hourly_snippet_chart.png')

plt.show()