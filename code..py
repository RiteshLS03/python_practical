
# # Experiment 10: Temperature Data Analysis and Prediction


# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Load the dataset (replace with correct path or URL if needed)
# url = "https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv"
# data = pd.read_csv(url)

# # Display basic information about the dataset
# print("Dataset Info:")
# print(data.info())

# # Display first few rows
# print("\nFirst 5 Rows:")
# print(data.head())

# # -----------------------------
# # Data Preprocessing
# data['Date'] = pd.to_datetime(data['Date'])
# data['Year'] = data['Date'].dt.year
# data['Month'] = data['Date'].dt.month

# # Check for missing values
# print("\nMissing Values:")
# print(data.isnull().sum())

# # -----------------------------
# # Data Visualization
# plt.figure(figsize=(12, 6))
# sns.lineplot(x='Date', y='Mean', data=data, label="Global Mean Temperature")
# plt.title('Global Mean Temperature Over Time')
# plt.xlabel('Year')
# plt.ylabel('Mean Temperature (°C)')
# plt.legend()
# plt.show()

# # -----------------------------
# # Feature Selection and Split
# data = data.dropna(subset=['Mean'])
# X = data[['Year', 'Month']]
# y = data['Mean']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # -----------------------------
# # Model Training - Linear Regression
# model = LinearRegression()
# model.fit(X_train, y_train)

# # -----------------------------
# # Model Evaluation
# y_pred = model.predict(X_test)
# print("\nModel Evaluation:")
# print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
# print(f"R-squared: {r2_score(y_test, y_pred):.4f}")

# # -----------------------------
# # Predict Future Temperatures
# future_years = np.array(range(2025, 2031))
# future_data = pd.DataFrame({'Year': np.repeat(future_years, 12), 'Month': list(range(1, 13)) * len(future_years)})
# future_data['Predicted_Temp'] = model.predict(future_data[['Year', 'Month']])

# # Plot predicted values
# plt.figure(figsize=(12, 6))
# sns.lineplot(x=future_data['Year'] + future_data['Month'] / 12, y=future_data['Predicted_Temp'], label="Predicted Temp")
# plt.title('Future Temperature Predictions')
# plt.xlabel('Year')
# plt.ylabel('Predicted Temperature (°C)')
# plt.legend()
# plt.show()

# print("\nFuture Predictions:")
# print(future_data[['Year', 'Month', 'Predicted_Temp']].head(12))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dummy global temperature data
np.random.seed(42)  # For reproducibility

# Create dummy dates from 1950 to 2024
start_date = pd.Timestamp('1950-01-01')
end_date = pd.Timestamp('2024-12-31')
dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly start

# Create a trend component (increasing temperatures)
years = (dates.year - dates.year.min()) / (dates.year.max() - dates.year.min())
trend = 0.8 * years  # 0.8°C overall warming

# Create seasonal component (monthly variations)
seasonal = 0.2 * np.sin(2 * np.pi * (dates.month - 1) / 12)

# Create random noise
noise = np.random.normal(0, 0.1, size=len(dates))

# Combine components
temperature = -0.3 + trend + seasonal + noise  # Starting around -0.3°C anomaly

# Create the DataFrame
data = pd.DataFrame({
    'Date': dates.strftime('%Y-%m-%d'),
    'Mean': temperature,
    'Land': temperature + np.random.normal(0.05, 0.05, size=len(dates)),
    'Ocean': temperature - np.random.normal(0.05, 0.05, size=len(dates)),
})

# Load the dataset (use our dummy data instead of the URL)
# url = "https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv"
# data = pd.read_csv(url)

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Display first few rows
print("\nFirst 5 Rows:")
print(data.head())

# -----------------------------
# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# -----------------------------
# Data Visualization
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Mean', data=data, label="Global Mean Temperature")
plt.title('Global Mean Temperature Over Time')
plt.xlabel('Year')
plt.ylabel('Mean Temperature Anomaly (°C)')
plt.legend()
plt.savefig('global_temp_trend.png')  # Save the figure
# plt.show()  # Commented out for artifact

# -----------------------------
# Feature Selection and Split
data = data.dropna(subset=['Mean'])
X = data[['Year', 'Month']]
y = data['Mean']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Model Training - Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Model Evaluation
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R-squared: {r2_score(y_test, y_pred):.4f}")

# -----------------------------
# Predict Future Temperatures
future_years = np.array(range(2025, 2031))
future_data = pd.DataFrame({'Year': np.repeat(future_years, 12), 
                           'Month': list(range(1, 13)) * len(future_years)})
future_data['Predicted_Temp'] = model.predict(future_data[['Year', 'Month']])

# Plot predicted values
plt.figure(figsize=(12, 6))
sns.lineplot(x=future_data['Year'] + future_data['Month'] / 12, 
            y=future_data['Predicted_Temp'], label="Predicted Temp")
plt.title('Future Temperature Predictions')
plt.xlabel('Year')
plt.ylabel('Predicted Temperature (°C)')
plt.legend()
plt.savefig('future_predictions.png')  # Save the figure
# plt.show()  # Commented out for artifact

print("\nFuture Predictions:")
print(future_data[['Year', 'Month', 'Predicted_Temp']].head(12))

# Create a csv file with the dummy data
data.to_csv('dummy_global_temperature_data.csv', index=False)
print("\nDummy data saved to 'dummy_global_temperature_data.csv'")