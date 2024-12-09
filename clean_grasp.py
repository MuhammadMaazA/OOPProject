import pandas as pd

# Load the data
csv_file = 'grasp_data_cylinder.csv'
data = pd.read_csv(csv_file)

# Remove the 'Step' column
if 'Step' in data.columns:
    data = data.drop(columns=['Step'])

# Add the 'Radius' column with a value of 0.22
data.insert(0, 'Radius', 0.22)  # Adding at the start of the dataframe

# Save the updated data to a new file
updated_csv_file = 'grasp_data_cylinder.csv'
data.to_csv(updated_csv_file, index=False)

print(
    f"Updated data without 'Step' and with 'Radius' column saved to {updated_csv_file}")
