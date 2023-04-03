import pandas as pd
from sklearn.ensemble import IsolationForest

def run_isolation_forest(model_data: pd.DataFrame, contamination=0.005, n_estimators=200,max_samples=0.7) -> pd.DataFrame:
   IF = (IsolationForest(random_state=0, contamination=contamination, n_estimators=n_estimators, max_samples=max_samples))
   IF.fit(model_data)
   output = pd.Series(IF.predict(model_data)).apply(lambda x: 1 if x == -1 else 0)
   score = IF.decision_function(model_data)
   return output, score



df = pd.read_csv("E:/Users/dromanow/PycharmProjects/AnomalyDetectionWithDropout/nyc_taxi.csv", parse_dates=['timestamp'])
df_hourly = df.set_index('timestamp').resample('H').mean().reset_index()
df_daily = df.set_index('timestamp').resample('D').mean().reset_index()
df_weekly = df.set_index('timestamp').resample('W').mean().reset_index()
for DataFrame in [df_hourly, df_daily]:
    DataFrame['Weekday'] = (pd.Categorical(DataFrame['timestamp'].dt.strftime('%A'),
                                           categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday', 'Saturday', 'Sunday'])
                           )
    DataFrame['Hour'] = DataFrame['timestamp'].dt.hour
    DataFrame['Day'] = DataFrame['timestamp'].dt.weekday
    DataFrame['Month'] = DataFrame['timestamp'].dt.month
    DataFrame['Year'] = DataFrame['timestamp'].dt.year
    DataFrame['Month_day'] = DataFrame['timestamp'].dt.day
    DataFrame['Lag'] = DataFrame['value'].shift(1)
    DataFrame['Rolling_Mean'] = DataFrame['value'].rolling(7, min_periods=1).mean()
    DataFrame = DataFrame.dropna()

df_hourly = (df_hourly
             .join(df_hourly.groupby(['Hour','Weekday'])['value'].mean(),
                   on = ['Hour', 'Weekday'], rsuffix='_Average')
            )

df_daily = (df_daily
            .join(df_daily.groupby(['Hour','Weekday'])['value'].mean(),
                  on = ['Hour', 'Weekday'], rsuffix='_Average')
           )

df_hourly.tail()

df_hourly.dropna(inplace=True)

# Daily
df_daily_model_data = df_daily[['value', 'Hour', 'Day',  'Month','Month_day','Rolling_Mean']].dropna()

# Hourly
model_data = df_hourly[['value', 'Hour', 'Day', 'Month_day', 'Month','Rolling_Mean','Lag', 'timestamp']].set_index('timestamp').dropna()
model_data.head()

# print(model_data)

outliers, score = run_isolation_forest(model_data)
df_hourly = (df_hourly.assign(Outliers = outliers).assign(Score = score))

print(df_hourly)

df_hourly.to_csv('nyc_taxi_labeled.csv')
def outliers(thresh):
    print(f'Number of Outliers below Anomaly Score Threshold {thresh}:')
    print(len(df_hourly.query(f"Outliers == 1 & Score <= {thresh}")))

outliers(0.05)