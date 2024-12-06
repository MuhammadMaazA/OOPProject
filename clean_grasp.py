import pandas as pd


csv_file = 'cleaned_grasp_data.csv'
data = pd.read_csv(csv_file)


data['Success'] = data['Delta Z'].apply(lambda x: 1 if x > 0.1 else (2 if 0.05 <= x <= 0.1 else 0))


updated_csv_file = 'updated_grasp_data.csv'
data.to_csv(updated_csv_file, index=False)

print(f"Updated data saved to {updated_csv_file}")
