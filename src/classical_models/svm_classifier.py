from sklearn.svm import SVC

from src.utils.data_utils import load_data, split_dataset

from src.utils.evaluation import (
    calculate_accuracy,
    print_confusion_matrix,
    print_classification_report,
    calculate_auc
)


def train_model(X_train, y_train):

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def make_predictions(model, X_test):

    predictions = model.predict(X_test)

    return predictions


def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = train_model(X_train, y_train)

    predictions = make_predictions(model, X_test)

    accuracy = calculate_accuracy(y_test, predictions)

    auc = calculate_auc(model, X_test, y_test)

    print("SVM Model Trained Successfully!")

    print(f"\nAccuracy: {accuracy:.4f}")

    print_confusion_matrix(y_test, predictions)

    print_classification_report(y_test, predictions)

    print(f"ROC-AUC Score: {auc:.4f}")


if __name__ == "__main__":
    main()

