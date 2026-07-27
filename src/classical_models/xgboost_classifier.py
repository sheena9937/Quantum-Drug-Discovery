from xgboost import XGBClassifier

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

    model = XGBClassifier(

        n_estimators=100,

        learning_rate=0.1,

        max_depth=6,

        random_state=42,

        eval_metric="logloss"

    )

    model.fit(X_train, y_train)

    return model

def make_predictions(model, X_test):

    predictions = model.predict(X_test)

    return predictions

def get_feature_importance(model):

    importance = model.feature_importances_

    return importance

import os
import joblib

def save_model(model):

    os.makedirs("../../results/models", exist_ok=True)

    joblib.dump(
        model,
        "../../results/models/xgboost_model.pkl"
    )

    print("Model saved successfully!")

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

    print("XGBoost Model Trained Successfully!")

    print(f"\nAccuracy: {accuracy:.4f}")

    print_confusion_matrix(y_test, predictions)

    print_classification_report(y_test, predictions)

    print(f"ROC-AUC Score: {auc:.4f}")


if __name__ == "__main__":
    main()



