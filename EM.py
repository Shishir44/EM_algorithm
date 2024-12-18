import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset provided by the user
file_path = '/mnt/data/Mall_Customers.csv'
data = pd.read_csv(file_path)

# Preprocessing the dataset
# Dropping non-numerical and unnecessary columns
data_preprocessed = data.drop(['CustomerID', 'Genre'], axis=1)

# Normalizing the numerical features for better clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_preprocessed)

# Converting back to a DataFrame for clarity
data_preprocessed_scaled = pd.DataFrame(data_scaled, columns=data_preprocessed.columns)

# Display the first few rows of the preprocessed data
def expectation_maximization_multivariate(data, num_components, max_iter=100, tol=1e-4):
    n, d = data.shape  # n = number of samples, d = number of dimensions
    weights = np.ones(num_components) / num_components  # Mixture weights
    means = data[np.random.choice(n, num_components, replace=False)]  # Random initial means
    covariances = np.array([np.cov(data, rowvar=False) + np.eye(d) for _ in range(num_components)])  # Initial covariances

    log_likelihood = -np.inf
    for iteration in range(max_iter):
        # E-Step: Compute responsibilities
        responsibilities = np.zeros((n, num_components))
        for k in range(num_components):
            responsibilities[:, k] = weights[k] * multivariate_normal.pdf(
                data, mean=means[k], cov=covariances[k], allow_singular=True
            )
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-Step: Update parameters
        N_k = responsibilities.sum(axis=0)
        weights = N_k / n
        means = (responsibilities.T @ data) / N_k[:, None]
        covariances = np.array([
            (responsibilities[:, k][:, None] * (data - means[k])).T @ (data - means[k]) / N_k[k] + np.eye(d) * 1e-6
            for k in range(num_components)
        ])

        # Compute log-likelihood
        new_log_likelihood = np.sum(np.log(responsibilities @ weights))

        # Check for convergence
        if np.abs(new_log_likelihood - log_likelihood) < tol:
            break
        log_likelihood = new_log_likelihood

    return weights, means, covariances

# Example usage
num_components = 2  # Assume 2 clusters for this example
data_np = data_preprocessed_scaled.to_numpy()
weights, means, covariances = expectation_maximization_multivariate(data_np, num_components)

# Print results
print("Mixture Weights:", weights)
print("Means:", means)
print("Covariances:", covariances)

# Step 4: Visualize Results
plt.scatter(data_np[:, 0], data_np[:, 1], alpha=0.5, label='Data')
for k in range(num_components):
    plt.scatter(means[k, 0], means[k, 1], label=f'Gaussian {k+1} Mean', s=100)
plt.title("Clustering with EM Algorithm")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.legend()
plt.show()
