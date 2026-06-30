from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def calculate_accuracy(y_true, predictions):

    return accuracy_score(y_true, predictions)

def print_confusion_matrix(y_true, predictions):

    matrix = confusion_matrix(y_true, predictions)

    print("\nConfusion Matrix:")

    print(matrix)

def print_classification_report(y_true, predictions):

    report = classification_report(y_true, predictions)

    print("\nClassification Report:")

    print(report)

def calculate_auc(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)

    return auc

