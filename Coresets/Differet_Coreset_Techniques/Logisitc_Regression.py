import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.linalg import qr

# Generate a synthetic dataset
def generate_data(n_samples=100000, n_features=20):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples) > 0.5).astype(int)
    return X, y

# Define the coreset methods
def kmeans_coreset(X, y, m):
    kmeans = KMeans(n_clusters=20, random_state=42).fit(X)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    coreset, weights, y_core = [], [], []

    for i in range(len(centers)):
        cluster_points = X[labels == i]
        cluster_labels = y[labels == i]
        cluster_size = len(cluster_points)
        sampled_indices = np.random.choice(cluster_size, size=min(m // len(centers), cluster_size), replace=False)
        coreset.append(cluster_points[sampled_indices])
        y_core.append(cluster_labels[sampled_indices])
        weights.append(np.ones(len(sampled_indices)) * (cluster_size / len(sampled_indices)))

    return np.vstack(coreset), np.hstack(weights), np.hstack(y_core)

def leverage_coreset(X, y, m):
    Q, _ = qr(X, mode='economic')
    leverage_scores = np.sum(Q ** 2, axis=1)
    probabilities = leverage_scores / np.sum(leverage_scores)
    sampled_indices = np.random.choice(len(X), size=m, replace=False, p=probabilities)
    return X[sampled_indices], probabilities[sampled_indices], y[sampled_indices]

def monotonic_coreset(X, y, m):
    row_norms = np.linalg.norm(X, axis=1)
    probabilities = row_norms / np.sum(row_norms)
    sampled_indices = np.random.choice(len(X), size=m, replace=False, p=probabilities)
    return X[sampled_indices], probabilities[sampled_indices], y[sampled_indices]

def lewis_coreset(X, y, m, t=5):
    Q, _ = qr(X, mode='economic')
    leverage_scores = np.sum(Q ** 2, axis=1)
    for _ in range(t):
        leverage_scores += np.sum(Q ** 2, axis=1)
    probabilities = leverage_scores / np.sum(leverage_scores)
    sampled_indices = np.random.choice(len(X), size=m, replace=False, p=probabilities)
    return X[sampled_indices], probabilities[sampled_indices], y[sampled_indices]

def osmac_coreset(X, y, m):
    model = LogisticRegression(max_iter=100).fit(X, y)
    residuals = np.abs(y - model.predict_proba(X)[:, 1])
    probabilities = residuals / np.sum(residuals)
    sampled_indices = np.random.choice(len(X), size=m, replace=False, p=probabilities)
    return X[sampled_indices], probabilities[sampled_indices], y[sampled_indices]

def uniform_coreset(X, y, m):
    sampled_indices = np.random.choice(len(X), size=m, replace=False)
    return X[sampled_indices], np.ones(m) / m, y[sampled_indices]

# Main script
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

methods = [
    ("kmeans", kmeans_coreset),
    ("leverage", leverage_coreset),
    ("monotonic", monotonic_coreset),
    ("lewis", lewis_coreset),
    ("osmac", osmac_coreset),
    ("uniform", uniform_coreset),
]

m = 100 # 10000 samples in the dataset reduced to 500 in the coreset
n_runs = 100

results = {method[0]: [] for method in methods}

# Evaluate each coreset method
for name, method in methods:
    for _ in range(n_runs):
        X_core, weights, y_core = method(X_train, y_train, m)
        model = LogisticRegression(max_iter=100).fit(X_core, y_core, sample_weight=weights)
        predictions = model.predict_proba(X_test)[:, 1]
        loss = log_loss(y_test, predictions)
        results[name].append(loss)

# Compute the loss on the full dataset
full_model = LogisticRegression(max_iter=100).fit(X_train, y_train)
full_predictions = full_model.predict_proba(X_test)[:, 1]
full_loss = log_loss(y_test, full_predictions)


# Print the results

print("=" * 80)
print("Coreset Method Comparison")
print("=" * 80)
print(f"Full Dataset Loss: {full_loss:.4f}")
for name, losses in results.items():
    print(f"{name}: {np.mean(losses):.4f} ± {np.std(losses):.4f}")

# Plot the results
plt.figure(figsize=(12, 8))

for name, losses in results.items():
    plt.plot(range(n_runs), losses, label=name)

# Mark the global minimum
all_losses = [(name, i, loss) for name, losses in results.items() for i, loss in enumerate(losses)]
global_min = min(all_losses, key=lambda x: x[2])
plt.scatter(global_min[1], global_min[2], color='red', label=f"Global Min: {global_min[0]} Loss = {global_min[2]:.4f}", zorder=5)

# Add a horizontal line for the full dataset loss
plt.axhline(y=full_loss, color='black', linestyle='--', label=f"Full Dataset Loss = {full_loss:.4f}")

plt.xlabel("Run Index")
plt.ylabel("Log-Loss")
plt.title("Coreset Method Comparison Across 100 Runs")
plt.legend()
plt.grid()
plt.show()