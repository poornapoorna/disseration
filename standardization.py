import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# List files in the specified directory
dir_path = '/content/drive/MyDrive/dataset'
print("Directory path:", dir_path)

# List the files in the directory
files = os.listdir(dir_path)
print("Files in directory:", files)

def standardize_csv_files(path, sep=','):
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]  # Only consider CSV files

    for csv_file in csv_files:
        try:
            file_path = os.path.join(path, csv_file)
            # Read the CSV file
            df = pd.read_csv(file_path, sep=sep)
            # Standardize the features
            scaler = StandardScaler()
            df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            # Print the standardized DataFrame
            print(f"\nStandardized data for {csv_file}:")
            print(df_standardized.head())
            # Save the standardized DataFrame to a new CSV file
            standardized_file_path = os.path.join(path, 'standardized_' + csv_file)
            df_standardized.to_csv(standardized_file_path, index=False)
            print(f"Standardized {csv_file} and saved to {standardized_file_path}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

# Standardize all CSV files in the directory
standardize_csv_files(dir_path, sep=',')  # Adjust 'sep' if needed

def view_standardized_files(path, sep=','):
    standardized_csv_files = [f for f in os.listdir(path) if f.startswith('standardized_') and f.endswith('.csv')]

    for standardized_csv_file in standardized_csv_files:
        try:
            standardized_file_path = os.path.join(path, standardized_csv_file)
            df_standardized = pd.read_csv(standardized_file_path, sep=sep)
            print(f"\nViewing standardized data for {standardized_csv_file}:")
            print(df_standardized.head())
        except Exception as e:
            print(f"Error viewing {standardized_csv_file}: {e}")

# View the standardized CSV files
view_standardized_files(dir_path, sep=',')  # Adjust 'sep' if needed
