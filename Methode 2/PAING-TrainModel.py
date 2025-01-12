"""
This script trains a Decision Tree classifier on the extracted features and labels.
The features and labels are created by using the scripts PAING-method2Mod.py and PAING-AssignLabels.py respectively.
The script uses cross-validation to evaluate the model and prints the classification report on the test set.

@Author: Thant Zin Htoo PAING
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the data
features_df = pd.read_csv('features.csv')
labels_df = pd.read_csv('images.csv')

# Ensure both datasets have the same number of rows
assert len(features_df) == len(labels_df)

# Merge datasets
data = pd.concat([features_df, labels_df['Label']], axis=1)

# Prepare the data
X = data[['Mean Kappa', 'Std Kappa', 'Mean Eigen', 'Std Eigen', 'Tampered Percentage']]
y = data['Label'].map({'Real': 0, 'Fake': 1})  # Encode labels

# Initialize the classifier
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=2)

# Initialize Stratified k-fold for cross-validation
k = 5  # Number of folds
skf = StratifiedKFold(n_splits=k)

# Cross-validation scores
cv_scores = cross_val_score(dt_model, X, y, cv=skf, scoring='accuracy')

print(f"Cross-Validation Scores for {k} Folds: {cv_scores}")
print(f"Average Accuracy: {cv_scores.mean():.2f}")

# Train and evaluate on the full dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt_model.fit(X_train, y_train)

# Test set predictions
y_pred = dt_model.predict(X_test)

# Print metrics
print("\nFinal Model Evaluation on Test Set")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))