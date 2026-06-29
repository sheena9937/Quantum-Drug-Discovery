import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


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

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model

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



def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = train_model(X_train, y_train)
    predictions = make_predictions(model, X_test)
    accuracy = evaluate_model(y_test, predictions)
    auc = calculate_auc(model, X_test, y_test)

    print("Model trained successfully!")

    print()

    print("Training Samples:", len(X_train))

    print("Testing Samples:", len(X_test))
    print("\nFirst 10 Predictions:")
    print(f"\nAccuracy: {accuracy:.4f}")
    print_confusion_matrix(y_test, predictions)
    print_classification_report(y_test, predictions)
    print(f"ROC-AUC Score: {auc:.4f}")



if __name__ == "__main__":
    main()