# OOP Project

This repository contains code for:
1. **Generating grasp data** using PyBullet simulations for two object types (**cuboid** or **cylinder**).
2. **Classifying** those grasps (using a **Random Forest** model).
3. **Visualizing** the resulting data in various plots and heatmaps.

---
## **Key Files**

- **`main.py`**  
  - **Top-level** script to either **generate new data** or **classify** existing data.
  - Runs a PyBullet **GUI** so you can see the simulation (when generating new data).

- **`train_model.py`**  
  - Trains a **Random Forest** classifier on either cuboid or cylinder data.
  - Saves the model in `models/` and updates the CSV with predictions.

- **`evaluate.py`**  
  - Optional evaluator logic (e.g., threshold-based success).

- **`robots/gripper.py`**  
  - Contains the `BaseGripper`, `PR2Gripper`, `CustomGripper` classes for PyBullet gripper control.

- **`plots/`**  
  - Python scripts to visualize or analyze results (e.g., heatmaps, confusion matrices, 3D scatter).

- **`data/`**  
  - Where CSV files of generated grasps are stored (e.g., `grasp_data_cuboid.csv`) plus updated CSVs with predictions (e.g. `updated_grasp_data_cuboid_with_predictions.csv`).

- **`models/`**  
  - Contains the trained Random Forest models (e.g. `cuboid_grasp_model.pkl`).

---

## **Installation and Requirements**

1. **Create** and **activate** a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate or on Windows: venv\Scripts\activate

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## **How to Run**


1. Generating New Data
When prompted, type y to generate new data.
Choose the object shape (cuboid or cylinder).
Enter the number of grasps (e.g., 50).
A PyBullet GUI will open, showing each grasp attempt in real time.
After data generation, the script asks if you want to run classification (calls train_model.py automatically).


2. Using Existing Data
When asked about generating new data, type n to use existing data.
Then enter the shape to classify (cuboid or cylinder).
The script reads data/grasp_data_{shape}.csv, trains a RandomForest, saves the model to models/{shape}_grasp_model.pkl, and updates data/updated_grasp_data_{shape}_with_predictions.csv.

**Classification Outputs**
Trained model: saved in models/{shape}_grasp_model.pkl.
Updated CSV: predictions stored in data/updated_grasp_data_{shape}_with_predictions.csv.
Console output includes cross-validation scores, final test accuracy. Plots like ROC and feature importance appear in pop-up windows.

**Visualization**
Several scripts under the plots/ folder provide different analyses:
KDE.py: Generates 2D KDE heatmaps for Position X vs. Position Y and Position X vs. Position Z.
plots.py / plots_cylinder.py: Confusion matrices, ROC curves, 3D scatter plots, etc.
visualize.py: 3D visualization of gripper poses and orientations, overlaying cuboid or cylinder boundaries.
For example, to run the KDE plots:
python plots/KDE.py
This loads updated_grasp_data_cuboid_with_predictions.csv from the data/ folder and displays KDE heatmaps in a window.

