import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

data = pd.read_csv(os.path.join(
    "data", "updated_grasp_data_cylinder_with_predictions.csv"))

# Relevant columns for success and pose
position_columns = ['Position X', 'Position Y', 'Position Z']
orientation_columns = ['Orientation Roll',
                       'Orientation Pitch', 'Orientation Yaw']
features = ['Radius', *position_columns, *orientation_columns]

# Plot 1: Success Rate vs. Amount of Data


def plot_success_rate_vs_data(data):
    num_points = range(1, len(data) + 1)
    success_rate = [data.iloc[:i]['Success'].mean() for i in num_points]

    plt.figure(figsize=(10, 6))
    plt.plot(num_points, success_rate, color='blue', label='Success Rate')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs. Number of Data Points (Cylinder)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Plot 2: Confusion Matrix for Classifier Evaluation


def plot_confusion_matrix(data):
    y_true = data['Success']
    y_pred = data['Predicted Success']

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                                  'Failure', 'Almost', 'Success'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Cylinder Grasp Classifier')
    plt.tight_layout()
    plt.show()

# Plot 3: Pose Variability in Successful Grasps


def plot_successful_grasps(data):
    successful_data = data[data['Success'] == 2]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(successful_data['Position X'], successful_data['Position Y'], successful_data['Position Z'],
               c='green', marker='o', label='Successful Grasps')

    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_zlabel('Position Z')
    ax.set_title('Pose Variability in Successful Grasps (Cylinder)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot 4: Sensitivity Analysis Plot


def plot_sensitivity_analysis(data):
    data['Roll Bin'] = pd.cut(data['Orientation Roll'],
                              bins=np.linspace(-np.pi, np.pi, 20))
    success_rate_per_bin = data.groupby('Roll Bin')['Success'].mean()

    plt.figure(figsize=(10, 6))
    success_rate_per_bin.plot(kind='bar', color='purple', alpha=0.7)
    plt.xlabel('Orientation Roll (Binned)')
    plt.ylabel('Average Success Rate')
    plt.title('Sensitivity Analysis of Grasp Success to Roll Angle (Cylinder)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Plot 5: Feature Importance


def plot_feature_importance(data):
    X = data[features]
    y = data['Success']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importance = model.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(features, importance, color='skyblue')
    plt.title('Feature Importance (Cylinder)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.grid(axis='x')
    plt.show()

# Plot 6: ROC Curve


def plot_roc_curve(data):
    X = data[features]
    y = data['Success']

    y_binarized = label_binarize(y, classes=[0, 1, 2])
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    y_score = model.predict_proba(X)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 6))
    for i, color in zip(range(3), ['blue', 'orange', 'green']):
        plt.plot(fpr[i], tpr[i],
                 label=f'Class {i} (AUC = {roc_auc[i]:.2f})', color=color)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve (Cylinder)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()


# Run the plots
plot_success_rate_vs_data(data)
plot_confusion_matrix(data)
plot_successful_grasps(data)
plot_sensitivity_analysis(data)
plot_feature_importance(data)
