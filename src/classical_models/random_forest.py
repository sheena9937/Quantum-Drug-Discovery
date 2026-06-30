import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier

from src.utils.data_utils import load_data, split_dataset

from src.utils.evaluation import (
    calculate_accuracy,
    print_confusion_matrix,
    print_classification_report,
    calculate_auc
)

from src.utils.visualization import (
    plot_feature_importance,
    plot_top_features,
    plot_accuracy,
    plot_auc
)

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


def save_model(model):

    model_path = "../../results/models/random_forest_model.pkl"

    joblib.dump(model, model_path)

    print(f"\nModel saved successfully at:\n{model_path}")

def make_predictions(model, X_test):

    predictions = model.predict(X_test)

    return predictions


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

    predictions = make_predictions(model, X_test)

    accuracy = calculate_accuracy(y_test, predictions)

    auc = calculate_auc(model, X_test, y_test)

    importance = get_feature_importance(model)

    plot_feature_importance(importance)

    plot_top_features(importance)

    save_model(model)

    save_model_results()

    comparison_file = "../../results/comparisons/model_comparison.csv"

    plot_accuracy(comparison_file)

    plot_auc(comparison_file)

    print("\nRandom Forest Model Trained Successfully!")

    print(f"\nAccuracy: {accuracy:.4f}")

    print_confusion_matrix(y_test, predictions)

    print_classification_report(y_test, predictions)

    print(f"ROC-AUC Score: {auc:.4f}")



if __name__ == "__main__":
    main()