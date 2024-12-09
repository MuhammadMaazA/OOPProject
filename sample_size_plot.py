import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the datasets for cuboid and cylinder
cuboid_data = pd.read_csv('updated_grasp_data_with_predictions.csv')
cylinder_data = pd.read_csv('updated_grasp_data_cylinder_with_predictions.csv')

# Function to compute accuracy for different sample sizes


def compute_accuracies(data, sample_sizes):
    X = data[['Radius', 'Position X', 'Position Y', 'Position Z',
              'Orientation Roll', 'Orientation Pitch', 'Orientation Yaw']]
    y = data['Success'].astype(int)

    accuracies = []

    for sample_size in sample_sizes:
        accuracies_per_sample = []

        # Repeat multiple times to get confidence intervals
        for _ in range(5):
            # Sample the data
            X_sample, _, y_sample, _ = train_test_split(
                X, y, train_size=sample_size, random_state=None)

            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_sample, y_sample, test_size=0.2, random_state=42)

            # Train the model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Evaluate accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies_per_sample.append(accuracy)

        # Store mean and std dev of accuracies
        accuracies.append((np.mean(accuracies_per_sample),
                          np.std(accuracies_per_sample)))

    return accuracies


# Define sample sizes
sample_sizes = range(100, min(len(cuboid_data), len(cylinder_data)), 100)

# Compute accuracies for both cuboid and cylinder datasets
cuboid_accuracies = compute_accuracies(cuboid_data, sample_sizes)
cylinder_accuracies = compute_accuracies(cylinder_data, sample_sizes)

# Extract means and standard deviations
cuboid_mean = [a[0] for a in cuboid_accuracies]
cuboid_std = [a[1] for a in cuboid_accuracies]
cylinder_mean = [a[0] for a in cylinder_accuracies]
cylinder_std = [a[1] for a in cylinder_accuracies]

# Plot the results
plt.figure(figsize=(10, 6))

# Plot cuboid data
plt.plot(sample_sizes, [m * 100 for m in cuboid_mean],
         label='Cuboid - Mean Accuracy (%)', color='blue')
plt.fill_between(sample_sizes,
                 [(m - s) * 100 for m, s in zip(cuboid_mean, cuboid_std)],
                 [(m + s) * 100 for m, s in zip(cuboid_mean, cuboid_std)],
                 color='blue', alpha=0.2, label='Cuboid Confidence Interval')

# Plot cylinder data
plt.plot(sample_sizes, [m * 100 for m in cylinder_mean],
         label='Cylinder - Mean Accuracy (%)', color='orange')
plt.fill_between(sample_sizes,
                 [(m - s) * 100 for m, s in zip(cylinder_mean, cylinder_std)],
                 [(m + s) * 100 for m, s in zip(cylinder_mean, cylinder_std)],
                 color='orange', alpha=0.2, label='Cylinder Confidence Interval')

# Customize the plot
plt.xlabel('Sample Size')
plt.ylabel('Mean Accuracy (%)')
plt.title('Impact of Sample Size on Classifier Performance - Cuboid vs Cylinder')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
