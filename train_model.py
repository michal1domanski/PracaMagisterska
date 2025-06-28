import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump

CSV_PATH = "C:/Users/Michał/Desktop/Praca-magisterska/PracaMagisterska/ML_model/best_lane_model.pkl.csv"
BEST_MODEL_OUTPUT = "C:/Users/Michał/Desktop/Praca-magisterska/PracaMagisterska/ML_model/best_lane_model.pkl"

def evaluate_model(name, model, X_test, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Object": model,
        "Prediction": y_pred
    }

def train_and_compare_models():
    df = pd.read_csv(CSV_PATH)
    X = df[['error']].values
    y = df['steer'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        # "Lasso": Lasso(),
        "SVR": SVR()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(name, model, X_test, y_test, y_pred)
        results.append(metrics)

    # Sort by RMSE
    results.sort(key=lambda r: r['RMSE'])

    # Display metrics
    print("📊 Model Comparison:")
    for r in results:
        print(f"\n{r['Model']}")
        print(f"  MSE:  {r['MSE']:.4f}")
        print(f"  RMSE: {r['RMSE']:.4f}")
        print(f"  MAE:  {r['MAE']:.4f}")
        print(f"  R²:   {r['R2']:.4f}")

    # 🎯 Save best model
    best_model = results[0]["Object"]
    dump(best_model, BEST_MODEL_OUTPUT)
    print(f"\n✅ Best model '{results[0]['Model']}' saved to {BEST_MODEL_OUTPUT}")

    # 📈 Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    for r in results:
        plt.scatter(X_test, r["Prediction"], alpha=0.5, label=r["Model"])
    plt.scatter(X_test, y_test, color='black', label='Actual', s=20)
    plt.xlabel('Lane Error')
    plt.ylabel('Steering')
    plt.title('Prledictions vs Actua')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 🧮 Plot prediction errors for best model
    best_pred = results[0]["Prediction"]
    errors = y_test - best_pred
    plt.hist(errors, bins=30, color='orange', edgecolor='black')
    plt.title(f'Prediction Errors: {results[0]["Model"]}')
    plt.xlabel('Error (y_true - y_pred)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def tune_models():
    df = pd.read_csv(CSV_PATH)
    X = df[['error']].values
    y = df['steer'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🎯 Ridge tuning
    ridge_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    ridge = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error')
    ridge.fit(X_train, y_train)
    print(f"🔧 Best Ridge alpha: {ridge.best_params_['alpha']}")

    # 🎯 SVR tuning
    svr_params = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']}
    svr = GridSearchCV(SVR(), svr_params, cv=5, scoring='neg_mean_squared_error')
    svr.fit(X_train, y_train)
    print(f"🔧 Best SVR C: {svr.best_params_['C']}")

if __name__ == "__main__":
    train_and_compare_models()
    tune_models()
