import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("weather.csv")


print(df.head())
print(df.info())
print(df.describe())

df['Date'] = pd.to_datetime(df['Date'])

df = df.dropna()


df = df[['Date', 'Temperature', 'Rainfall', 'Humidity']]

daily_mean = np.mean(df['Temperature'])
daily_min  = np.min(df['Temperature'])
daily_max  = np.max(df['Temperature'])
daily_std  = np.std(df['Temperature'])

print("Daily Mean Temperature:", daily_mean)
print("Daily Min Temperature:", daily_min)
print("Daily Max Temperature:", daily_max)
print("Daily Std Temperature:", daily_std)

df['Month'] = df['Date'].dt.month
df['Year']  = df['Date'].dt.year

plt.figure()
plt.plot(df['Date'], df['Temperature'])
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Daily Temperature Trend")
plt.savefig("daily_temperature.png")
plt.show()

monthly_rain = df.groupby('Month')['Rainfall'].sum()

plt.figure()
plt.bar(monthly_rain.index, monthly_rain.values)
plt.xlabel("Month")
plt.ylabel("Rainfall")
plt.title("Monthly Rainfall")
plt.savefig("monthly_rainfall.png")
plt.show()

plt.figure()
plt.scatter(df['Temperature'], df['Humidity'])
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Humidity vs Temperature")
plt.savefig("humidity_vs_temperature.png")
plt.show()

plt.figure(figsize=(8,5))

plt.subplot(2,1,1)
plt.plot(df['Date'], df['Temperature'])
plt.title("Daily Temperature")

plt.subplot(2,1,2)
plt.scatter(df['Temperature'], df['Humidity'])
plt.title("Temp vs Humidity")

plt.tight_layout()
plt.savefig("combined_plot.png")
plt.show()

# Group by month and find average temperature
month_group = df.groupby('Month')['Temperature'].mean()
print("Average Temperature by Month:")
print(month_group)

df.to_csv("cleaned_weather.csv", index=False)

