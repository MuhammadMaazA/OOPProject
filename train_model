import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('updated_grasp_data.csv')

# Update feature columns to match CSV file columns
X = data[['Position X', 'Position Y', 'Position Z', 'Orientation Roll', 'Orientation Pitch', 'Orientation Yaw']]
y = data['Success']

# Convert target labels to categorical if necessary
y = y.astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Model training completed. Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, 'grasp_model.pkl')

# Predict on the entire dataset to add predictions
data['Predicted Success'] = model.predict(X)

# Save the updated dataset with predictions
data.to_csv('updated_grasp_data_with_predictions.csv', index=False)
