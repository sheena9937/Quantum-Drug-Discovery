from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from src.utils.data_utils import load_data, split_dataset

from src.utils.evaluation import (
    calculate_accuracy,
    print_confusion_matrix,
    print_classification_report,
    calculate_auc
)

def apply_smote(X_train, y_train):

    smote = SMOTE(
        random_state=42
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    return X_train_smote, y_train_smote

def train_model(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000,
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

    print("Before SMOTE")

    print("Training samples:", len(y_train))

    print("Class 0:", sum(y_train == 0))

    print("Class 1:", sum(y_train == 1))

    print()

    X_train, y_train = apply_smote(
        X_train,
        y_train
    )

    print("After SMOTE")

    print("Training samples:", len(y_train))

    print("Class 0:", sum(y_train == 0))

    print("Class 1:", sum(y_train == 1))

    print()

    model = train_model(X_train, y_train)

    predictions = make_predictions(model, X_test)

    accuracy = calculate_accuracy(y_test, predictions)

    auc = calculate_auc(model, X_test, y_test)

    print("Logistic Regression with SMOTE")

    print(f"\nAccuracy: {accuracy:.4f}")

    print_confusion_matrix(y_test, predictions)

    print_classification_report(y_test, predictions)

    print(f"ROC-AUC Score: {auc:.4f}")


if __name__ == "__main__":
    main()
