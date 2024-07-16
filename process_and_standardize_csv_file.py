import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_and_standardize_csv_files(path, sep=','):
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    standardizer = StandardScaler()
    
    for csv_file in csv_files:
        file_path = os.path.join(path, csv_file)
        try:
            # Load the data
            df = pd.read_csv(file_path, sep=sep)
            print(f"Loaded {csv_file} successfully.")

            # Standardize the data
            df_standardized = pd.DataFrame(standardizer.fit_transform(df.dropna()), columns=df.columns)
            print(f"Standardized {csv_file}.")

            # Save the standardized data
            standardized_file_path = os.path.join(path, 'standardized_' + csv_file)
            df_standardized.to_csv(standardized_file_path, index=False)
            print(f"Saved standardized data to {standardized_file_path}.")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

# Directory path
dir_path = '/content/drive/MyDrive/dataset'
print("Directory path:", dir_path)

# Process and Standardize CSV files
process_and_standardize_csv_files(dir_path)
