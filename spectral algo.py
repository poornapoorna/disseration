import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Assuming the heart dataset is already loaded into a DataFrame named `df`
df = dataframes_dict['heart']

# Select the features to standardize
features_to_standardize = df.columns

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
df_standardized = pd.DataFrame(scaler.fit_transform(df[features_to_standardize]), columns=features_to_standardize)

# Convert DataFrame to numpy array for further processing
X = df_standardized.values

# Step 1: Construct the affinity matrix
sigma = 1
A = -1 * np.square(X[:, None, :] - X[None, :, :]).sum(axis=-1)
A = np.exp(A / (2 * sigma**2))
np.fill_diagonal(A, 0)


# Step 3: Create graph and visualize
A1 = np.copy(A)
A1[A1 < 0.9] = 0
G = nx.from_numpy_array(A1)

# Use spring_layout to generate 2D positions for visualization
pos = nx.spring_layout(G)

plt.figure(figsize=(6, 6))
plt.axis('off')
nx.draw_networkx_nodes(G, pos=pos, node_size=20, alpha=0.9)
nx.draw_networkx_edges(G, pos=pos, edge_color="white", alpha=0.3)
plt.show()

# Step 4: Compute the graph Laplacian
I = np.zeros_like(A)
np.fill_diagonal(I, 1)

D = np.zeros_like(A)
np.fill_diagonal(D, np.sum(A, axis=1))
D_inv_sqrt = np.linalg.inv(np.sqrt(D))

L = I - np.dot(D_inv_sqrt, A).dot(D_inv_sqrt)

# Step 5: Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(L)
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

# Order the eigenvalues in an increasing order
ind = np.argsort(eigenvalues, axis=0)
eigenvalues_sorted = np.take_along_axis(eigenvalues, ind, axis=0)

# Order the eigenvectors based on the magnitude of their corresponding eigenvalues
eigenvectors_sorted = eigenvectors.take(ind, axis=1)

# Step 6: Visualize the eigenvectors
fig, axs = plt.subplots(3, 3, figsize=(16, 12))
eigen_v_x = np.linspace(0, eigenvectors_sorted.shape[0], eigenvectors_sorted.shape[0])

for j, ax in enumerate(fig.axes):
    if j < len(eigenvalues_sorted):
        eigen_v_y = eigenvectors_sorted[:, j]
        ax.scatter(eigen_v_x, eigen_v_y, marker='o')
        ax.set_title(f'eigenvector {j} | eigenvalue: {eigenvalues_sorted[j]:.4f}')

plt.show()

# Step 7: Select eigenvectors for clustering
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(eigenvectors_sorted[:, 0], eigenvectors_sorted[:, 3], marker='o', s=40)

ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
               labelbottom=False, labeltop=False, labelleft=False, labelright=False)
ax.set(xlabel=None, ylabel=None)
plt.show()

# Use the eigenvectors corresponding to the smallest non-zero eigenvalues
X_transformed = eigenvectors_sorted[:, [0, 3]]

# Step 8: Standardize the transformed data
scaler = StandardScaler()
X_transformed_scaled = scaler.fit_transform(X_transformed)

# Step 9: Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X_transformed_scaled)

# Step 10: Visualize the true and predicted labels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], marker='o', s=40)
ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
ax1.set(xlabel=None, ylabel=None)
ax1.set_title('True labels')

ax2.scatter(X[:, 0], X[:, 1], marker='o', s=40, c=kmeans.labels_, cmap='viridis')
ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
ax2.set(xlabel=None, ylabel=None)
ax2.set_title('Predicted labels')

plt.show()
