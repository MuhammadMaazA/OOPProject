"""
train_model.py

Trains a Random Forest model on either cuboid or cylinder data in 'data' folder,
saves the model to 'models/{shape}_grasp_model.pkl', and updates CSV with predictions.

Columns (old format) expected in CSV:
Position X,Position Y,Position Z,Orientation Roll,Orientation Pitch,Orientation Yaw,
Initial Z,Final Z,Delta Z,Success
"""

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc


def train_model(shape):
    """
    Train a Random Forest for 'cuboid' or 'cylinder'.
    Reads data/grasp_data_{shape}.csv from data folder,
    Splits into train/test,
    Cross-validates, final fit,
    Saves model to models/{shape}_grasp_model.pkl,
    Writes updated CSV (data/updated_grasp_data_{shape}_with_predictions.csv).
    Plots ROC and feature importances.
    """
    shape = shape.lower()
    if shape not in ["cuboid", "cylinder"]:
        print("[ERROR] Shape must be 'cuboid' or 'cylinder'.")
        return

    data_folder = "data"
    csv_file = os.path.join(data_folder, f"grasp_data_{shape}.csv")
    if not os.path.isfile(csv_file):
        print(
            f"[ERROR] {csv_file} does not exist. Please generate data first.")
        return

    print(f"[INFO] Loading dataset from {csv_file}")
    data = pd.read_csv(csv_file)

    if "Success" not in data.columns:
        print("[ERROR] 'Success' column not found in CSV. Aborting.")
        return

    # old format columns as features
    features = [
        "Position X", "Position Y", "Position Z",
        "Orientation Roll", "Orientation Pitch", "Orientation Yaw"
    ]
    for col in features:
        if col not in data.columns:
            print(f"[ERROR] Column '{col}' missing. Aborting.")
            return

    X = data[features]
    y = data["Success"].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"[INFO] Cross-validation scores for {shape}: {cv_scores}")
    print(
        f"[INFO] Mean cross-validation accuracy: {cv_scores.mean() * 100:.2f}%")

    # Fit final model
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"[INFO] Final test accuracy ({shape}): {accuracy * 100:.2f}%")

    # Save model in 'models' folder
    models_folder = "models"
    os.makedirs(models_folder, exist_ok=True)
    model_file = os.path.join(models_folder, f"{shape}_grasp_model.pkl")
    joblib.dump(model, model_file)
    print(f"[INFO] Saved model to {model_file}")

    # Append predicted successes to entire dataset
    data['Predicted Success'] = model.predict(X)

    # Write updated CSV
    updated_csv = os.path.join(
        data_folder, f"updated_grasp_data_{shape}_with_predictions.csv")
    data.to_csv(updated_csv, index=False)
    print(f"[INFO] Wrote updated dataset with predictions to {updated_csv}")

    # ROC curve
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({shape.capitalize()})')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Feature importances
    importances = model.feature_importances_
    plt.figure()
    plt.bar(X.columns, importances, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importance ({shape.capitalize()})')
    plt.tight_layout()
    plt.show()
