# Coreset for Machine Learning Tasks

This repository contains Python scripts for generating and evaluating coresets for machine learning tasks, including:

1. **Linear Regression**
2. **Logistic Regression**
3. **Support Vector Machines (SVM)**


## Files

### 1. `linear_regression.py`
This script generates coresets for linear regression tasks. It uses leverage scores to sample and weight subsets of data, ensuring that the linear regression model trained on the coreset closely approximates the model trained on the full dataset.

**Features:**
- Generates leverage score-based coresets.
- Compares model errors between full dataset and coresets.
- Performs 1000 trials to identify the best coreset.

### 2. `logistic_regression.py`
This script focuses on logistic regression tasks. It generates coresets and evaluates the log loss on both the full dataset and the coresets.

**Features:**
- Balanced binary classification dataset generation.
- Leverage score sampling for coresets.
- Visual comparison of log loss for 1000 coreset trials.

### 3. `svm.py`
This script applies coreset sampling to support vector machine (SVM) classification tasks. It evaluates and compares the error (1 - accuracy) on the full dataset and coresets.

**Features:**
- Binary classification dataset generation.
- 1000 coreset trials to determine the best subset.
- Plots the error across all trials, highlighting the best coreset.


## Each script:
- Generates synthetic data.
- Produces coresets using leverage scores.
- Compares model performance on the full dataset and coresets.
- Visualizes the results using a plot.

For example:
- **`linear_regression.py`** compares errors between the full dataset and coresets.
- **`logistic_regression.py`** evaluates the log loss for the full dataset and coresets.
- **`svm.py`** analyzes and compares classification errors (1 - accuracy) for the full dataset and coresets.
