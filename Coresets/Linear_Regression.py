import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class CoresetGenerator:
    def __init__(self, sampling_ratio=0.1):
        """
        Initialize the Coreset Generator.
        
        Parameters:
        sampling_ratio (float): Proportion of the dataset to include in the coreset.
        """
        if not 0 < sampling_ratio <= 1:
            raise ValueError("Sampling ratio must be between 0 and 1.")
        self.sampling_ratio = sampling_ratio

    def generate_coreset(self, X, y):
        """
        Generate a coreset for a linear regression task.
        
        Parameters:
        X (np.ndarray): Feature matrix (n_samples, n_features).
        y (np.ndarray): Target values (n_samples,).
        
        Returns:
        X_coreset (np.ndarray): Feature matrix of the coreset.
        y_coreset (np.ndarray): Target values of the coreset.
        weights (np.ndarray): Weights for the coreset samples.
        """
        n_samples = X.shape[0]
        if n_samples != len(y):
            raise ValueError("X and y must have the same number of samples.")

        # Compute leverage scores
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)
        residuals = y - reg.predict(X)
        leverage_scores = np.sum((X @ np.linalg.pinv(X.T @ X) * X), axis=1)

        # Normalize leverage scores to form probabilities
        probabilities = leverage_scores / np.sum(leverage_scores)

        # Select samples based on leverage score probabilities
        sample_size = max(1, int(self.sampling_ratio * n_samples))
        selected_indices = np.random.choice(n_samples, size=sample_size, replace=False, p=probabilities)

        # Generate coreset
        X_coreset = X[selected_indices]
        y_coreset = y[selected_indices]
        weights = 1 / (probabilities[selected_indices] * sample_size)

        return X_coreset, y_coreset, weights

if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    true_coeffs = np.random.rand(10)
    y = X @ true_coeffs + np.random.randn(1000) * 0.1  # Add some noise

    # Perform 1000 random samplings
    num_trials = 1000
    best_coreset_error = float('inf')
    best_X_coreset = None
    best_y_coreset = None
    best_weights = None
    coreset_errors = []

    coreset_generator = CoresetGenerator(sampling_ratio=0.1)
    
    full_model = LinearRegression().fit(X, y)
    y_pred_full = full_model.predict(X)
    full_error = mean_squared_error(y, y_pred_full)

    for _ in range(num_trials):
        X_coreset, y_coreset, weights = coreset_generator.generate_coreset(X, y)
        coreset_model = LinearRegression().fit(X_coreset, y_coreset, sample_weight=weights)
        y_pred_coreset = coreset_model.predict(X)
        coreset_error = mean_squared_error(y, y_pred_coreset)
        coreset_errors.append(coreset_error)

        if coreset_error < best_coreset_error:
            best_coreset_error = coreset_error
            best_X_coreset = X_coreset
            best_y_coreset = y_coreset
            best_weights = weights

    print("Original data size:", X.shape)
    print("Coreset size:", best_X_coreset.shape)
    print("Full data error:", full_error)
    print("Best coreset error:", best_coreset_error)
    print("Relative increase in error:", round(((best_coreset_error - full_error) / full_error * 100), 2), "%") # negative sign because error should be increasing

    # Plot the errors
    plt.figure(figsize=(10, 6))
    plt.plot(coreset_errors, label="Coreset Errors", alpha=0.7)
    plt.axhline(y=full_error, color='r', linestyle='--', label="Full Data Error")
    best_trial_index = np.argmin(coreset_errors)
    plt.scatter(best_trial_index, best_coreset_error, color='g', s=100, label="Best Coreset Error")
    plt.title("Coreset Errors vs Full Data Error")
    plt.xlabel("Trial Index")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()


