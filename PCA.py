import os
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load standardized CSV files
def load_standardized_files(path):
    standardized_csv_files = [f for f in os.listdir(path) if f.startswith('standardized_') and f.endswith('.csv')]
    dataframes = {}

    for standardized_csv_file in standardized_csv_files:
        try:
            standardized_file_path = os.path.join(path, standardized_csv_file)
            df_standardized = pd.read_csv(standardized_file_path)
            dataframes[standardized_csv_file] = df_standardized
        except Exception as e:
            print(f"Error loading {standardized_csv_file}: {e}")

    return dataframes

# Perform PCA
def perform_pca(dataframes):
    for file_name, df in dataframes.items():
        print(f"\nProcessing {file_name} for PCA...")

        # Assuming the first column is an identifier or should be excluded
        features = df  # Adjust column index as needed

        # Standardizing is assumed done previously
        pca = PCA()
        pca.fit(features)

        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()

        eigenvalues = pca.explained_variance_
        # Print summary
        summary_df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(len(explained_variance))],
            'Standard Deviation': pca.singular_values_,
            'Proportion of Variance': explained_variance,
            'Cumulative Proportion': cumulative_variance
        })
        print(summary_df)
        eigenvalue_df = pd.DataFrame({
            'Eigenvalue': eigenvalues,
            'Explained Variance': explained_variance,
            'Cumulative Variance': explained_variance.cumsum()
        })
        print(f"\nEigenvalues and Variance for {file_name}:\n", eigenvalue_df)
        # Scree plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual explained variance')
        plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o', color='red', label='Cumulative explained variance')
        plt.title(f'PCA Explained Variance for {file_name}')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance')
        plt.xticks(range(1, len(explained_variance) + 1))
        plt.legend()
        plt.grid()
        plt.show()
         # Variable contributions
        var_contributions = pca.components_**2
        cos2 = var_contributions / np.sum(var_contributions, axis=1)[:, np.newaxis]

        # Create a DataFrame for easier visualization
        contributions_df = pd.DataFrame(cos2, index=[f'PC{i+1}' for i in range(len(cos2))], columns=df.columns)


        # Correlation plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(var_contributions, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Variable Contributions for {file_name}')
        plt.show()

# Main Execution
dir_path = '/content/drive/MyDrive/dataset'
standardized_dataframes = load_standardized_files(dir_path)
perform_pca(standardized_dataframes)
