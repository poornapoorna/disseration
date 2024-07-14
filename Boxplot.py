import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Specify the directory containing the datasets
dir_path = '/content/drive/MyDrive/dataset'

#Files in directory: ['heart.csv', 'breast tissue.csv', 'liver.csv', 'dermatology_database_1.csv', 'kidney_disease.csv', 'fetal_health.csv', 'breast cancer wisconsin.csv', 'diabetes.csv', '.DS_Store']
# Dictionary to map file names to target variables to remove


# Iterate through each file in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.csv'):  # Filter for CSV files
        filepath = os.path.join(dir_path, filename)

        # 1. Load the data from the CSV file
        data = pd.read_csv(filepath)


        # 3. Reshape the data to long format (include all columns)
        data_long = data.melt(var_name='Variable', value_name='Value')

        # 4. Create the boxplot (using seaborn)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Value', y='Variable', data=data_long, palette="Set3")
        plt.title(f'Boxplot of All Variables ({filename})')  # Include filename in title
        plt.show()
