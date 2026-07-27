from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from src.utils.data_utils import load_data, split_dataset

from src.utils.evaluation import (
    calculate_accuracy,
    print_confusion_matrix,
    print_classification_report,
    calculate_auc
)

def load_dataset():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    return X_train, X_test, y_train, y_test

def create_model():

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    return model

def get_parameter_grid():

    parameter_grid = {

        "n_estimators": [50, 100, 150],

        "max_depth": [3, 5, 7],

        "learning_rate": [0.01, 0.1, 0.2]

    }

    return parameter_grid

def perform_grid_search(model, parameter_grid, X_train, y_train):

    grid_search = GridSearchCV(

        estimator=model,

        param_grid=parameter_grid,

        cv=5,

        scoring="roc_auc",

        n_jobs=-1

    )

    grid_search.fit(X_train, y_train)

    return grid_search

def main():

    X_train, X_test, y_train, y_test = load_dataset()

    model = create_model()

    parameter_grid = get_parameter_grid()

    grid_search = perform_grid_search(
        model,
        parameter_grid,
        X_train,
        y_train
    )

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross Validation ROC-AUC:")
    print(f"{grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy = calculate_accuracy(y_test, predictions)

    auc = calculate_auc(best_model, X_test, y_test)

    print(f"\nTest Accuracy: {accuracy:.4f}")

    print_confusion_matrix(y_test, predictions)

    print_classification_report(y_test, predictions)

    print(f"Test ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()