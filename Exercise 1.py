import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_olivetti_faces

# Load the Olivetti faces dataset
olivetti = fetch_olivetti_faces(shuffle=True, random_state=42)
X = olivetti.data

# Step 1: Dimensionality Reduction with PCA
pca = PCA(n_components=0.99, svd_solver='full')
X_reduced = pca.fit_transform(X)

# Step 2: Determine the most suitable covariance_type
covariance_types = ['full', 'tied', 'diag', 'spherical']
bic_scores = []
aic_scores = []

for cov_type in covariance_types:
    gmm = GaussianMixture(n_components=10, covariance_type=cov_type, random_state=42)
    gmm.fit(X_reduced)
    bic_scores.append(gmm.bic(X_reduced))
    aic_scores.append(gmm.aic(X_reduced))

# Plot BIC and AIC scores for different covariance types
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(covariance_types, bic_scores, marker='o', linestyle='-', color='b', label='BIC')
plt.title('BIC Scores for Different Covariance Types')
plt.xlabel('Covariance Type')
plt.ylabel('BIC Score')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(covariance_types, aic_scores, marker='o', linestyle='-', color='r', label='AIC')
plt.title('AIC Scores for Different Covariance Types')
plt.xlabel('Covariance Type')
plt.ylabel('AIC Score')
plt.legend()

# Step 3: Determine the minimum number of clusters
n_components_range = range(1, 21)
aic_scores = []
bic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_reduced)
    aic_scores.append(gmm.aic(X_reduced))
    bic_scores.append(gmm.bic(X_reduced))

# Plot AIC and BIC scores for different numbers of clusters
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(n_components_range, bic_scores, marker='o', linestyle='-', color='b', label='BIC')
plt.title('BIC Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC Score')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_components_range, aic_scores, marker='o', linestyle='-', color='r', label='AIC')
plt.title('AIC Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('AIC Score')
plt.legend()

plt.tight_layout()
plt.show()

# Train a Gaussian Mixture Model
n_components = 10  # You can adjust this as needed
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(X_reduced)

# Predict the cluster assignments (hard clustering) for each instance
cluster_assignments = gmm.predict(X_reduced)

# Display the cluster assignments for each instance
for instance_index, cluster in enumerate(cluster_assignments):
    print(f"Instance {instance_index} is assigned to cluster {cluster}")

# Generate new faces using the GMM
n_samples = 5  # Number of new faces to generate
samples, _ = gmm.sample(n_samples)

# Transform the generated samples back to the original space
generated_faces = pca.inverse_transform(samples)

# Visualize the generated faces
def plot_faces(images, h, w, n_row=1, n_col=n_samples):
    plt.figure(figsize=(2 * n_col, 2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(f"Generated Face {i + 1}", size=12)
        plt.xticks(())
        plt.yticks(())

plot_faces(generated_faces, h=64, w=64)
plt.suptitle("Generated Faces", size=16)
plt.show()

# Define a function to modify images
def modify_images(images):
    modified_images = []
    for image in images:
        # Rotate the image by 90 degrees
        rotated_image = np.rot90(image.reshape(64, 64), k=1).ravel()
        
        # Flip the image horizontally
        flipped_image = np.fliplr(image.reshape(64, 64)).ravel()
        
        # Darken the image
        darkened_image = (image * 0.5).clip(0, 1)
        
        modified_images.extend([rotated_image, flipped_image, darkened_image])
    return modified_images

# Modify the generated faces
modified_generated_faces = modify_images(generated_faces)

# Visualize the modified faces
def plot_faces(images, h, w, n_row=1, n_col=n_samples * 3):
    plt.figure(figsize=(2 * n_col, 2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(f"Modified Face {i + 1}", size=12)
        plt.xticks(())
        plt.yticks(())

plot_faces(modified_generated_faces, h=64, w=64, n_row=n_samples, n_col=3)
plt.suptitle("Modified Faces", size=16)
plt.show()

# Train the GMM model again with the modified generated faces

# Compute log likelihood scores for normal images
normal_scores = gmm.score_samples(X_reduced)

# Compute log likelihood scores for anomalies
anomaly_scores = gmm.score_samples(pca.transform(modified_generated_faces))

# Define a threshold for anomaly detection (you can adjust this threshold)
threshold = -8.0

# Determine anomalies by comparing the scores to the threshold
anomalies = np.where(anomaly_scores < threshold)[0]

print(f"Anomalies detected at indices: {anomalies}")
