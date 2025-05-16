import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump
import matplotlib.pyplot as plt

CSV_PATH = "C:/Users/Michał/Desktop/Praca-magisterska/PracaMagisterska/ML_model/lane_steering_model.pkl.csv"  # Change as needed
MODEL_OUTPUT = "C:/Users/Michał/Desktop/Praca-magisterska/PracaMagisterska/ML_model/lane_steering_model.pkl"

def train_model():
    df = pd.read_csv(CSV_PATH)
    X = df[['error']].values
    y = df['steer'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Ridge()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ Model trained. MSE: {mse:.4f}")

    dump(model, MODEL_OUTPUT)
    print(f"💾 Model saved to {MODEL_OUTPUT}")

    # Optional: visualize
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted', alpha=0.5)
    plt.xlabel('Lane Error')
    plt.ylabel('Steering')
    plt.legend()
    plt.title('Lane Error vs Steering Prediction')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_model()
