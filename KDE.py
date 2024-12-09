
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cuboid and cylinder data
cuboid_data = pd.read_csv('updated_grasp_data_with_predictions.csv')
cylinder_data = pd.read_csv('updated_grasp_data_cylinder_with_predictions.csv')

# Generate KDE plots for X-Y and X-Z planes for both datasets


def generate_kde_heatmap(data, x_col, y_col, title):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=data, x=x_col, y=y_col, fill=True,
                cmap="coolwarm", cbar=True, bw_adjust=0.5)
    plt.title(title, fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.tight_layout()
    plt.close()


# Generate KDE plots for cuboid
generate_kde_heatmap(cuboid_data, 'Position X', 'Position Y',
                     'KDE Heatmap of Grasp Positions (X-Y) - Cuboid')
generate_kde_heatmap(cuboid_data, 'Position X', 'Position Z',
                     'KDE Heatmap of Grasp Positions (X-Z) - Cuboid')

# Generate KDE plots for cylinder
generate_kde_heatmap(cylinder_data, 'Position X', 'Position Y',
                     'KDE Heatmap of Grasp Positions (X-Y) - Cylinder')
generate_kde_heatmap(cylinder_data, 'Position X', 'Position Z',
                     'KDE Heatmap of Grasp Positions (X-Z) - Cylinder')
