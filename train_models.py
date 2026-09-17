import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/assam_prepared_dataset_final_v2.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("TRAINING FINAL ASSAM TRANSPORT DELAY MODELS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

TARGET = "actual_arrival_delay_min"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Feature count: {X.shape[1]}")
print(f"Target: {TARGET}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# DEFINE MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

trained_models = {}

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SELECT MODEL
# ============================================================

# Select the model with the lowest MAE.
best_model_name = results_df.loc[
    results_df["MAE"].idxmin(),
    "model"
]

best_model = trained_models[best_model_name]

print("\n" + "=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print(f"Model selected: {best_model_name}")


# ============================================================
# SAVE BEST MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "final_assam_delay_model_v2.pkl"
)

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)


# ============================================================
# SAVE MODEL COLUMNS
# ============================================================

columns_path = os.path.join(
    MODEL_DIR,
    "final_assam_model_columns_v2.pkl"
)

with open(columns_path, "wb") as f:
    pickle.dump(list(X.columns), f)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {
    "model_name": best_model_name,
    "target": TARGET,
    "feature_count": X.shape[1],
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "random_state": 42,
    "dataset": DATA_PATH,
    "metrics": results_df.to_dict(orient="records")
}

info_path = os.path.join(
    MODEL_DIR,
    "final_assam_model_info_v2.pkl"
)

with open(info_path, "wb") as f:
    pickle.dump(model_info, f)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

comparison_path = os.path.join(
    MODEL_DIR,
    "final_model_comparison_v2.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(f"\nBest model : {best_model_name}")

print("\nSaved files:")

print(f"1. {model_path}")
print(f"2. {columns_path}")
print(f"3. {info_path}")
print(f"4. {comparison_path}")

print("\nValidation:")

if len(results_df) == 4:
    print("✓ 4 models trained successfully")

if X.shape[1] == 61:
    print("✓ 61 features confirmed")

if len(df) == 5000:
    print("✓ 5000 dataset rows confirmed")

print("\nStep 8 completed successfully.")