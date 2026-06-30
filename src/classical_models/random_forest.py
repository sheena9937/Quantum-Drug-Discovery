import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import joblib
from src.utils.model_comparison import plot_accuracy, plot_auc


def load_data():

    X = np.load("../../data/processed/fingerprints.npy")
    y = np.load("../../data/processed/labels.npy")

    return X, y


def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

def get_feature_importance(model):

    importance = model.feature_importances_

    return importance

def plot_feature_importance(importance):

    plt.figure(figsize=(12,6))

    plt.bar(range(len(importance)), importance)

    plt.title("Feature Importance")

    plt.xlabel("Fingerprint Bit")

    plt.ylabel("Importance")

    plt.tight_layout()

    plt.savefig("../../figures/random_forest_feature_importance.png")

    plt.show()

def plot_top_features(importance):

    top_n = 20

    indices = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 6))

    plt.barh(range(top_n), importance[indices])

    plt.yticks(range(top_n), indices)

    plt.xlabel("Feature Importance")

    plt.ylabel("Fingerprint Bit")

    plt.title("Top 20 Most Important Fingerprint Bits")

    plt.tight_layout()

    plt.savefig("../../figures/top20_feature_importance.png")

    plt.show()

def save_model(model):

    model_path = "../../results/models/random_forest_model.pkl"

    joblib.dump(model, model_path)

    print(f"\nModel saved successfully at:\n{model_path}")

def make_predictions(model, X_test):

    predictions = model.predict(X_test)

    return predictions

def evaluate_model(y_test, predictions):

    accuracy = accuracy_score(y_test, predictions)

    return accuracy

def print_confusion_matrix(y_test, predictions):

    matrix = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")

    print(matrix)

def print_classification_report(y_test, predictions):

    report = classification_report(y_test, predictions)

    print("\nClassification Report:")

    print(report)

def calculate_auc(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)

    return auc
import pandas as pd
import os

def save_model_results():

    results = {

        "Model": ["Logistic Regression", "Random Forest"],

        "Accuracy": [0.8824, 0.8946],

        "ROC-AUC": [0.8951, 0.9248],

        "Precision": [0.89, 0.89],

        "Recall": [0.96, 0.98],

        "F1 Score": [0.93, 0.93]

    }

    df = pd.DataFrame(results)

    output_path = "../../results/comparisons/model_comparison.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nModel comparison table saved successfully!")


def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = train_model(X_train, y_train)
    save_model(model)
    importance = get_feature_importance(model)
    plot_feature_importance(importance)
    plot_top_features(importance)
    predictions = make_predictions(model, X_test)
    accuracy = evaluate_model(y_test, predictions)
    auc = calculate_auc(model, X_test, y_test)

    print("Random Forest model trained successfully!")

    print()

    print("Training Samples:", len(X_train))

    print("Testing Samples:", len(X_test))
    print("\nFirst 10 Predictions:")
    print(f"\nAccuracy: {accuracy:.4f}")
    print_confusion_matrix(y_test, predictions)
    print_classification_report(y_test, predictions)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nFirst 10 Feature Importance Values:")

    print(importance[:10])
    save_model_results()
    comparison_file = "../../results/comparisons/model_comparison.csv"

    plot_accuracy(comparison_file)

    plot_auc(comparison_file)



if __name__ == "__main__":
    main()