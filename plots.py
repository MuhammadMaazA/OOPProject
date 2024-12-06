import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the data
data = pd.read_csv('updated_grasp_data_with_predictions.csv')

# Relevant columns for success and pose
position_columns = ['Position X', 'Position Y', 'Position Z']
orientation_columns = ['Orientation Roll', 'Orientation Pitch', 'Orientation Yaw']

# Plot 1: Success Rate vs. Amount of Data
def plot_success_rate_vs_data(data):
    # Group by the number of data points (cumulative)
    num_points = range(1, len(data) + 1)
    success_rate = [data.iloc[:i]['Success'].mean() for i in num_points]

    plt.figure(figsize=(10, 6))
    plt.plot(num_points, success_rate, color='blue', label='Success Rate')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs. Number of Data Points')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Plot 2: Confusion Matrix for Classifier Evaluation
def plot_confusion_matrix(data):
    # Assume 'Predicted Success' is a column containing model predictions
    y_true = data['Success']
    y_pred = data['Predicted Success']

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Failure', 'Almost', 'Success'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Grasp Classifier')
    plt.tight_layout()
    plt.show()

# Plot 3: Pose Variability in Successful Grasps
def plot_successful_grasps(data):
    # Only select successful grasps
    successful_data = data[data['Success'] == 2]
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(successful_data['Position X'], successful_data['Position Y'], successful_data['Position Z'],
               c='green', marker='o', label='Successful Grasps')
    
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_zlabel('Position Z')
    ax.set_title('Pose Variability in Successful Grasps')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot 4: Sensitivity Analysis Plot
def plot_sensitivity_analysis(data):
    # Calculate average success rate for different orientation roll bins
    data['Roll Bin'] = pd.cut(data['Orientation Roll'], bins=np.linspace(-np.pi, np.pi, 20))
    success_rate_per_bin = data.groupby('Roll Bin')['Success'].mean()
    
    plt.figure(figsize=(10, 6))
    success_rate_per_bin.plot(kind='bar', color='purple', alpha=0.7)
    plt.xlabel('Orientation Roll (Binned)')
    plt.ylabel('Average Success Rate')
    plt.title('Sensitivity Analysis of Grasp Success to Roll Angle')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Run the plots
plot_success_rate_vs_data(data)
plot_confusion_matrix(data)
plot_successful_grasps(data)
plot_sensitivity_analysis(data)
