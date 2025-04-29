# Code for Google Colab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lightgbm import LGBMRegressor

import warnings

warnings.filterwarnings('ignore')

# Seed
np.random.seed(42)


# 1. Weather Data (simulate OR fetch from API)
def get_weather_data(api_key=None, location="New York", simulate=True):
    if simulate or api_key is None:
        conditions = ['Clear', 'Rain', 'Snow', 'Fog', 'Hot', 'Cold']
        weather = {
            "temp": np.random.uniform(-5, 40),
            "weather_main": np.random.choice(conditions)
        }
        return weather
    else:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather = {
                "temp": data['main']['temp'],
                "weather_main": data['weather'][0]['main']
            }
            return weather
        else:
            print("Weather API fetch failed, using simulated data.")
            return get_weather_data(simulate=True)


# 2. Traffic Data
def simulate_traffic(hour=None):
    if hour is None:
        hour = datetime.now().hour
    if 7 <= hour <= 9 or 16 <= 18:
        return 3  # Heavy traffic
    elif 10 <= hour <= 15:
        return 2  # Moderate traffic
    else:
        return 1  # Light traffic


# 3. Road Condition Simulation
def simulate_road_condition(weather_main):
    if weather_main in ['Rain', 'Snow']:
        return 3  # Poor
    else:
        return np.random.choice([1, 2], p=[0.8, 0.2])  # Mostly good


# 4. Simulate Vehicle Data
def generate_vehicle_data(n_samples=1500):
    data = pd.DataFrame()

    # Core vehicle parameters
    data['engine_temp'] = np.random.normal(90, 8, n_samples)
    data['fuel_level'] = np.random.normal(50, 15, n_samples)
    data['battery_health'] = np.random.normal(85, 10, n_samples)
    data['mileage'] = np.random.normal(25000, 7000, n_samples)
    data['vehicle_age'] = np.random.normal(5, 2, n_samples)
    data['sensor_fault'] = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])

    # Scenario Columns
    data['road_type'] = np.random.choice(['Urban', 'Highway', 'Off-Road'], size=n_samples, p=[0.5, 0.4, 0.1])
    data['driving_style'] = np.random.choice(['Aggressive', 'Smooth', 'Stop-and-Go'], size=n_samples, p=[0.3, 0.4, 0.3])

    # External environment - weather, traffic, road condition
    weather_data = [get_weather_data(simulate=True) for _ in range(n_samples)]
    data['weather_temp'] = [w['temp'] for w in weather_data]
    data['weather_condition'] = [w['weather_main'] for w in weather_data]
    data['traffic_level'] = [simulate_traffic() for _ in range(n_samples)]
    data['road_condition'] = [simulate_road_condition(w['weather_main']) for w in weather_data]

    # --- Adjustments ---
    data.loc[data['road_type'] == 'Urban', 'battery_health'] -= np.random.normal(3, 1,
                                                                                 (data['road_type'] == 'Urban').sum())
    data.loc[data['road_type'] == 'Off-Road', 'sensor_fault'] = 1
    data.loc[data['road_type'] == 'Off-Road', 'engine_temp'] += np.random.normal(5, 2, (
                data['road_type'] == 'Off-Road').sum())

    data.loc[data['driving_style'] == 'Aggressive', 'engine_temp'] += np.random.normal(10, 3, (
                data['driving_style'] == 'Aggressive').sum())
    data.loc[data['driving_style'] == 'Aggressive', 'battery_health'] -= np.random.normal(5, 2, (
                data['driving_style'] == 'Aggressive').sum())

    data.loc[data['driving_style'] == 'Stop-and-Go', 'engine_temp'] += np.random.normal(5, 2, (
                data['driving_style'] == 'Stop-and-Go').sum())

    rain_snow = data['weather_condition'].isin(['Rain', 'Snow'])
    fog = data['weather_condition'] == 'Fog'
    hot = data['weather_condition'] == 'Hot'
    cold = data['weather_condition'] == 'Cold'

    data.loc[rain_snow, 'battery_health'] -= np.random.normal(4, 1, rain_snow.sum())
    data.loc[hot, 'engine_temp'] += np.random.normal(7, 2, hot.sum())
    data.loc[cold, 'battery_health'] -= np.random.normal(6, 2, cold.sum())
    data.loc[fog, 'sensor_fault'] = 1

    # --- Target Variable (time_until_service) ---
    base_service = (45000 - data['mileage']) / 120
    temp_effect = (100 - data['engine_temp']) * 0.12
    battery_effect = (100 - data['battery_health']) * 0.15
    traffic_effect = data['traffic_level'] * 2
    road_effect = data['road_condition'] * 3
    fault_penalty = data['sensor_fault'] * 25

    style_penalty = np.where(data['driving_style'] == 'Aggressive', 15,
                             np.where(data['driving_style'] == 'Stop-and-Go', 10, 0))

    weather_penalty = np.where(data['weather_condition'].isin(['Rain', 'Snow', 'Fog']), 10, 0)

    noise = np.random.normal(0, 6, n_samples)

    data['time_until_service'] = np.clip(
        base_service - temp_effect - battery_effect - traffic_effect - road_effect
        - fault_penalty - style_penalty - weather_penalty + noise,
        0, 365
    )

    return data


# Feature Engineering
def feature_engineering(df):
    df['engine_battery_ratio'] = df['engine_temp'] / (df['battery_health'] + 1)
    df['age_mileage_ratio'] = df['vehicle_age'] / (df['mileage'] + 1)
    df['fuel_engine_ratio'] = df['fuel_level'] / (df['engine_temp'] + 1)
    df['traffic_road_impact'] = df['traffic_level'] * df['road_condition']
    return df


# Preprocessing
def preprocess_data(df):
    X = df.drop('time_until_service', axis=1)
    y = df['time_until_service']

    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

    return preprocessor, X, y


# Train the Model
def train_model(preprocessor, X_train, y_train):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LGBMRegressor(random_state=42))
    ])

    param_grid = {
        'regressor__num_leaves': [31, 50, 70],
        'regressor__max_depth': [5, 7, 9],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__n_estimators': [150, 250, 350],
        'regressor__subsample': [0.7, 0.8],
        'regressor__colsample_bytree': [0.7, 0.9]
    }

    search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=3, verbose=1, n_jobs=-1,
                                scoring='neg_mean_squared_error')
    search.fit(X_train, y_train)

    print(f"Best Parameters Found: {search.best_params_}")
    return search.best_estimator_


# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nEvaluation Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([0, 365], [0, 365], '--r')
    plt.xlabel('Actual Time Until Service')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.grid()
    plt.show()


# Live Prediction
def live_prediction(model, api_key, location):
    weather = get_weather_data(api_key, location, simulate=False)
    traffic = simulate_traffic()
    road_condition = simulate_road_condition(weather['weather_main'])

    print("\n--- Enter Vehicle Info ---")
    engine_temp = float(input("Engine Temperature (°C): "))
    fuel_level = float(input("Fuel Level (%): "))
    battery_health = float(input("Battery Health (%): "))
    mileage = float(input("Mileage (km): "))
    vehicle_age = float(input("Vehicle Age (years): "))
    sensor_fault = int(input("Any sensor fault? (0 = No, 1 = Yes): "))
    road_type = input("Road Type (Urban/Highway/Off-Road): ")
    driving_style = input("Driving Style (Aggressive/Smooth/Stop-and-Go): ")

    sample = pd.DataFrame([{
        "engine_temp": engine_temp,
        "fuel_level": fuel_level,
        "battery_health": battery_health,
        "mileage": mileage,
        "vehicle_age": vehicle_age,
        "sensor_fault": sensor_fault,
        "road_type": road_type,
        "driving_style": driving_style,
        "weather_temp": weather['temp'],
        "weather_condition": weather['weather_main'],
        "traffic_level": traffic,
        "road_condition": road_condition
    }])

    sample = feature_engineering(sample)
    pred = model.predict(sample)[0]
    print(f"\nPredicted Time Until Next Service: {pred:.1f} days")


# Full Pipeline
def main():
    data = generate_vehicle_data(1500)
    data = feature_engineering(data)

    preprocessor, X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(preprocessor, X_train, y_train)

    evaluate_model(model, X_test, y_test)

    # Live prediction
    use_live = input("\nDo you want to predict live data? (yes/no): ").strip().lower()
    if use_live == "yes":
        api_key = input("Enter your OpenWeatherMap API key: ")
        location = input("Enter location for live weather (e.g., New York): ")
        live_prediction(model, api_key, location)


if __name__ == "__main__":
    main()
