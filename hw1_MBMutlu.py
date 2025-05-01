import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
# Load dataset
train_features = pd.read_csv("X_train.csv").to_numpy()
train_labels = pd.read_csv("y_train.csv", header=None).to_numpy().flatten()
test_features = pd.read_csv("X_test.csv").to_numpy()
test_labels = pd.read_csv("y_test.csv", header=None).to_numpy().flatten()

# Identify categorical columns
cat_columns = []
for i in range(train_features.shape[1]):
    if isinstance(train_features[0, i], str):
        cat_columns.append(i)

# Create encoding for categorical values
cat_mappings = {}
for col in cat_columns:
    unique_vals = np.unique(train_features[:, col])
    cat_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}

# Apply encoding to train and test sets
for col in cat_columns:
    vectorized_map = np.vectorize(lambda x: cat_mappings[col].get(x, -1))
    train_features[:, col] = vectorized_map(train_features[:, col])
    test_features[:, col] = vectorized_map(test_features[:, col])

# Convert to float type for consistency
train_features = train_features.astype(float)
test_features = test_features.astype(float)

feature_labels = ["Redshift", "Alpha", "Delta", "Green", "NearIR", "Cosmic", "Red", "UV", "IR"]

class SimpleNaiveBayes:
    def __init__(self):
        self.priors = {}
        self.feature_probs = defaultdict(lambda: defaultdict(dict))

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            self.priors[cls] = np.sum(y == cls) / len(y)
            subset = X[y == cls]
            for col in range(X.shape[1]):
                value_counts = {val: np.sum(subset[:, col] == val) for val in np.unique(subset[:, col])}
                for val, count in value_counts.items():
                    self.feature_probs[cls][col][val] = count / len(subset)

    def predict(self, X):
        import math
        predictions = []
        for sample in X:
            scores = {cls: np.log(self.priors[cls]) for cls in self.classes}
            for cls in self.classes:
                for col in range(X.shape[1]):
                    val = sample[col]
                    prob = self.feature_probs[cls][col].get(val, 1e-6)
                    scores[cls] += math.log(prob)
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)

# Train Naive Bayes
nb_model = SimpleNaiveBayes()
nb_model.fit(train_features, train_labels)
predictions = nb_model.predict(test_features)
accuracy = np.mean(predictions == test_labels)

# Confusion matrix
predictions = predictions.astype(int)
test_labels = test_labels.astype(int)
conf_matrix = np.zeros((2, 2), dtype=int)
for actual, predicted in zip(test_labels, predictions):
    if actual in [0, 1] and predicted in [0, 1]:
        conf_matrix[actual][predicted] += 1

print("'c:\\Users\\mbmut\\OneDrive\\Masaüstü\\intro to ml hw1\\hw1_deneme6.py'")
print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Hybrid Model
class HybridNaiveBayes(SimpleNaiveBayes):
    def fit(self, X, y):
        dependent_cols = [0, 3, 4, 6]
        super().fit(X, y)
        self.joint_probs = defaultdict(dict)
        for cls in self.classes:
            subset = X[y == cls]
            unique_combinations, counts = np.unique(subset[:, dependent_cols], axis=0, return_counts=True)
            for combo, count in zip(unique_combinations, counts):
                self.joint_probs[cls][tuple(combo)] = count / len(subset)

    def predict(self, X):
        import math
        predictions = []
        dependent_cols = [0, 3, 4, 6]
        for sample in X:
            scores = {cls: np.log(self.priors[cls]) for cls in self.classes}
            for cls in self.classes:
                for col in range(X.shape[1]):
                    if col in dependent_cols:
                        combo_vals = tuple(sample[dependent_cols])
                        prob = self.joint_probs[cls].get(combo_vals, 1e-6)
                        scores[cls] += math.log(prob)
                        break  
                for col in range(X.shape[1]):
                    if col not in dependent_cols:
                        val = sample[col]
                        prob = self.feature_probs[cls][col].get(val, 1e-6)
                        scores[cls] += math.log(prob)
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)

hybrid_model = HybridNaiveBayes()
hybrid_model.fit(train_features, train_labels)
hybrid_predictions = hybrid_model.predict(test_features)
hybrid_accuracy = np.mean(hybrid_predictions == test_labels)
print("Hybrid Model Accuracy:", hybrid_accuracy)

# Feature Importance (Mutual Information)
def calculate_mi(X, y):
    import math
    mi_values = np.zeros(X.shape[1])
    important_feature_values = {}
    for col in range(X.shape[1]):
        unique_x = np.unique(X[:, col])
        mi = 0
        for x_val in unique_x:
            for c_val in [0, 1]:
                joint_prob = np.sum((X[:, col] == x_val) & (y == c_val)) / len(y)
                if joint_prob > 0:
                    px = np.sum(X[:, col] == x_val) / len(y)
                    py = np.sum(y == c_val) / len(y)
                    mi += joint_prob * np.log(joint_prob / (px * py))
        mi_values[col] = mi
        # top value for class=1
        top_value = float(max(unique_x, key=lambda x: np.sum((X[:, col] == x) & (y == 1))))
        important_feature_values[col] = top_value
        # Print debugging line:
        print(f"{feature_labels[col]} MI Score: {mi:.6f}, Top Contributing Value: {top_value}")
    return mi_values, important_feature_values

mi_scores, important_feature_values = calculate_mi(train_features, train_labels)
top_features = [idx for idx, _ in sorted(enumerate(mi_scores), key=lambda x: -x[1])[:3]]
top_feature_names = [feature_labels[idx] for idx in top_features]
top_feature_values = [important_feature_values[idx] for idx in top_features]

print("Top 3 Important Features:", top_feature_names)
print("Top 3 Important Feature Values:", [float(v) for v in top_feature_values])

# Find top 3 feature values indicating NOT a Galaxy
top_feature_values_not_galaxy = []
for idx in top_features:
    feature_values_for_not_galaxy = train_features[train_labels == 0, idx]
    if len(feature_values_for_not_galaxy) > 0:
        most_common_value = max(
            set(feature_values_for_not_galaxy),
            key=list(feature_values_for_not_galaxy).count
        )
        top_feature_values_not_galaxy.append(most_common_value)
    else:
        top_feature_values_not_galaxy.append(-1)

your_not_galaxy_feature_values = pd.Series(
    top_feature_values_not_galaxy,
    index=["Very High redshift", "Sagittarius alpha", "Taurus alpha"],
    dtype="float64"
)
print("Your Top 3 Feature Values Indicating NOT a Galaxy:\n", your_not_galaxy_feature_values)

# Find top 3 feature values indicating a Galaxy
top_feature_values_galaxy = []
for idx in top_features:
    feature_values_for_galaxy = train_features[train_labels == 1, idx]
    if len(feature_values_for_galaxy) > 0:
        most_common_value = max(
            set(feature_values_for_galaxy),
            key=list(feature_values_for_galaxy).count
        )
        top_feature_values_galaxy.append(most_common_value)
    else:
        top_feature_values_galaxy.append(-1)

your_galaxy_feature_values = pd.Series(
    top_feature_values_galaxy,
    index=["Low redshift", "Medium redshift", "U_27 ultraviolet_filter"],
    dtype="float64"
)
print("Your Top 3 Feature Values Indicating a Galaxy:\n", your_galaxy_feature_values)


sorted_features = sorted(enumerate(mi_scores), key=lambda x: -x[1])
top_feature_indices = [idx for idx, _ in sorted_features]
top_feature_names = [feature_labels[idx] for idx in top_feature_indices]

# Function to evaluate Naïve Bayes with selected top-k features
def evaluate_with_k_features(k):
    selected_indices = top_feature_indices[:k]
    
    X_train_k = train_features[:, selected_indices]
    X_test_k = test_features[:, selected_indices]

    model = SimpleNaiveBayes()
    model.fit(X_train_k, train_labels)
    predictions = model.predict(X_test_k)

    # Compute Precision: TP / (TP + FP)
    TP = np.sum((predictions == 1) & (test_labels == 1))
    FP = np.sum((predictions == 1) & (test_labels == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision

# Evaluate precision for different values of k
k_values = range(1, len(feature_labels) + 1)
precisions = [evaluate_with_k_features(k) for k in k_values]

# Plot Precision vs. Number of Features (k)
plt.figure(figsize=(8, 5))
plt.plot(k_values, precisions, marker='o', linestyle='-')
plt.xlabel("Number of Selected Features (k)")
plt.ylabel("Precision")
plt.title("Effect of Feature Selection on Precision")
plt.grid()
plt.show()
