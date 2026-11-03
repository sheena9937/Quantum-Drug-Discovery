
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def main():
    root = Path(__file__).resolve().parents[2]

    data_dir = root / "data" / "processed"
    results_dir = root / "results" / "comparisons"
    figures_dir = root / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / "fingerprints.npy")
    y = np.load(data_dir / "labels.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, _, y_train, _ = train_test_split(
        X_train,
        y_train,
        train_size=120,
        random_state=42,
        stratify=y_train
    )

    X_test, _, y_test, _ = train_test_split(
        X_test,
        y_test,
        train_size=100,
        random_state=42,
        stratify=y_test
    )

    print("Training samples:", len(y_train))
    print("Testing samples:", len(y_test))

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "SVM": SVC(
            probability=True,
            random_state=42
        )
    }

    rows = []

    for name, model in models.items():
        print("\nTraining:", name)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(
            y_test,
            predictions
        )
        auc = roc_auc_score(y_test, probabilities)

        print("Accuracy:", accuracy)
        print("Balanced Accuracy:", balanced_acc)
        print("ROC-AUC:", auc)

        rows.append({
            "Model": name,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_acc,
            "ROC-AUC": auc
        })

    quantum_results = [
        {
            "Model": "Z Feature Map QSVC",
            "Accuracy": 0.77,
            "Balanced Accuracy": 0.5779,
            "ROC-AUC": 0.7708
        },
        {
            "Model": "ZZ QSVC",
            "Accuracy": 0.76,
            "Balanced Accuracy": 0.6140,
            "ROC-AUC": 0.7253
        },
        {
            "Model": "ZZ QSVC (2 Reps)",
            "Accuracy": 0.72,
            "Balanced Accuracy": 0.6447,
            "ROC-AUC": 0.7160
        },
        {
            "Model": "Original VQC",
            "Accuracy": 0.65,
            "Balanced Accuracy": 0.4419,
            "ROC-AUC": 0.6754
        },
        {
            "Model": "Balanced VQC",
            "Accuracy": 0.61,
            "Balanced Accuracy": 0.4298,
            "ROC-AUC": 0.4161
        }
    ]

    rows.extend(quantum_results)

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        results_dir / "final_model_comparison.csv",
        index=False
    )

    print("\nFINAL MODEL COMPARISON")
    print(comparison.to_string(index=False))

    metrics = [
        "Accuracy",
        "Balanced Accuracy",
        "ROC-AUC"
    ]

    for metric in metrics:
        plt.figure(figsize=(11, 6))

        plt.bar(
            comparison["Model"],
            comparison[metric]
        )

        plt.title(f"Model Comparison - {metric}")
        plt.ylabel(metric)
        plt.xlabel("Models")
        plt.ylim(0, 1)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        filename = metric.lower().replace(" ", "_")

        plt.savefig(
            figures_dir / f"{filename}_comparison.png",
            dpi=300
        )

        plt.close()

    print("\nComparison CSV and charts saved successfully.")


if __name__ == "__main__":
    main()
