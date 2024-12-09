import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc
import joblib
import matplotlib.pyplot as plt

# Load the dataset for cylinder grasp data
data = pd.read_csv('grasp_data_cylinder.csv')

# Define features excluding 'Radius'
X = data[['Position X', 'Position Y', 'Position Z',
          'Orientation Roll', 'Orientation Pitch', 'Orientation Yaw']]
y = data['Success'].astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the model with cross-validation
model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean() * 100:.2f}%")

# Fit the model
model.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Final test accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, 'cylinder_grasp_model.pkl')

# Predict on the entire dataset to add predictions
data['Predicted Success'] = model.predict(X)

# Save the updated dataset with predictions
data.to_csv('updated_grasp_data_cylinder_with_predictions.csv', index=False)

# Generate ROC curve
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(
    fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Cylinder)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Feature importance
feature_importances = model.feature_importances_
plt.figure()
plt.bar(X.columns, feature_importances, color='skyblue')
plt.xticks(rotation=45, ha='right')  # Rotate feature names to avoid overlap
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance (Cylinder)')
plt.tight_layout()
plt.show()
