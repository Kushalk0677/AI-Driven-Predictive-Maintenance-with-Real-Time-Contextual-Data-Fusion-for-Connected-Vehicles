# Code for Google Colab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

np.random.seed(42)

# Step 1: Simulate Realistic Vehicle Data
def generate_vehicle_data(n_samples=1000):
    data = pd.DataFrame({
        'engine_temp': np.random.normal(90, 10, n_samples),
        'fuel_level': np.random.normal(50, 10, n_samples),
        'battery_health': np.random.normal(80, 15, n_samples),
        'mileage': np.random.normal(20000, 5000, n_samples),
        'vehicle_age': np.random.normal(5, 2, n_samples),
        'sensor_fault': np.random.choice([0, 1], size=n_samples)
    })

    # Generate meaningful label based on realistic conditions
    conditions = (
        (data['engine_temp'] > 95) |
        (data['battery_health'] < 60) |
        (data['sensor_fault'] == 1) |
        (data['vehicle_age'] > 7) |
        (data['mileage'] > 30000)
    )
    data['maintenance_due'] = conditions.astype(int)
    return data

# Step 2: Feature Engineering
def feature_engineering(df):
    df['engine_battery_health'] = df['engine_temp'] * df['battery_health']
    df['age_mileage_ratio'] = df['vehicle_age'] / (df['mileage'] + 1)
    df['fuel_temp_ratio'] = df['fuel_level'] / (df['engine_temp'] + 1)
    return df

# Step 3: Preprocessing
def preprocess_data(df):
    X = df.drop('maintenance_due', axis=1)
    y = df['maintenance_due']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    return X_scaled, y_res

# Step 4: Train Models with Hyperparameter Tuning
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "Support Vector Machine": SVC(class_weight='balanced', probability=True, random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }

    param_grids = {
        "Logistic Regression": {'C': [0.1, 1, 10], 'penalty': ['l2']},
        "Random Forest": {'n_estimators': [100], 'max_depth': [10, 20, None]},
        "Support Vector Machine": {'C': [1], 'kernel': ['rbf']},
        "XGBoost": {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [6]},
        "LightGBM": {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [6]}
    }

    best_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        grid = GridSearchCV(model, param_grids[name], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_

    return best_models

# Step 5: Evaluate Models
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))
        results[name] = report

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {name}")
        plt.colorbar()
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.yticks([0, 1], ['No', 'Yes'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    return results

# Step 6: Random Forest Feature Importance
def feature_importance(model, X_train, feature_names):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print("\nFeature Importance (Random Forest):\n", feat_df)

    plt.figure(figsize=(8, 5))
    plt.barh(feat_df['Feature'], feat_df['Importance'])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance")
    plt.show()

# Step 7: Run everything
def main():
    data = generate_vehicle_data(1200)
    data = feature_engineering(data)
    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_models = train_models(X_train, y_train)
    results = evaluate_models(best_models, X_test, y_test)

    feature_importance(best_models["Random Forest"], X_train, data.drop('maintenance_due', axis=1).columns)
    return results

if __name__ == "__main__":
    results = main()
