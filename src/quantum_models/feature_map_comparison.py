
import numpy as np
import csv

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from qiskit.circuit.library import (
    z_feature_map,
    zz_feature_map
)

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


def load_data(data_dir):

    X_train = np.load(data_dir / "quantum_X_train.npy")
    X_test = np.load(data_dir / "quantum_X_test.npy")

    y_train = np.load(data_dir / "quantum_y_train.npy")
    y_test = np.load(data_dir / "quantum_y_test.npy")

    # Same pilot training subset as previous experiments
    X_train_small, _, y_train_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=120,
        random_state=42,
        stratify=y_train
    )

    # Same pilot testing subset as previous experiments
    X_test_small, _, y_test_small, _ = train_test_split(
        X_test,
        y_test,
        train_size=100,
        random_state=42,
        stratify=y_test
    )

    return (
        X_train_small,
        X_test_small,
        y_train_small,
        y_test_small
    )


def evaluate_feature_map(
        name,
        feature_map,
        X_train,
        X_test,
        y_train,
        y_test,
        results_dir
):

    print("\n" + "=" * 50)
    print("Experiment:", name)
    print("=" * 50)

    print("\nFeature map:")
    print(feature_map)

    # Quantum kernel
    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    # Balanced QSVC
    model = QSVC(
        quantum_kernel=quantum_kernel,
        class_weight="balanced"
    )

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Training completed.")

    # Predictions
    predictions = model.predict(X_test)

    decision_scores = model.decision_function(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(y_test, decision_scores)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        output_dict=True,
        zero_division=0
    )

    print("\nAccuracy:", accuracy)
    print("Balanced Accuracy:", balanced_accuracy)
    print("ROC-AUC:", auc)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            zero_division=0
        )
    )

    # Save individual predictions
    np.save(
        results_dir / f"{name}_predictions.npy",
        predictions
    )

    # Save individual metrics
    np.savez(
        results_dir / f"{name}_metrics.npz",
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        roc_auc=auc,
        confusion_matrix=cm
    )

    return {
        "Experiment": name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "ROC-AUC": auc,
        "Class 0 Precision": report["0"]["precision"],
        "Class 0 Recall": report["0"]["recall"],
        "Class 0 F1": report["0"]["f1-score"],
        "Class 1 Precision": report["1"]["precision"],
        "Class 1 Recall": report["1"]["recall"],
        "Class 1 F1": report["1"]["f1-score"]
    }


def main():

    project_root = Path(__file__).resolve().parents[2]

    data_dir = project_root / "data" / "processed"

    results_dir = project_root / "results" / "comparisons"

    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(data_dir)

    print("Training samples:", len(y_train))
    print("Testing samples:", len(y_test))

    print("\nTraining class distribution:")
    print(np.unique(y_train, return_counts=True))

    # Define feature-map experiments
    experiments = [
        (
            "Z_Feature_Map",
            z_feature_map(
                feature_dimension=4,
                reps=1
            )
        ),
        (
            "ZZ_Feature_Map",
            zz_feature_map(
                feature_dimension=4,
                reps=1,
                entanglement="linear"
            )
        ),
        (
            "ZZ_Feature_Map_2_Reps",
            zz_feature_map(
                feature_dimension=4,
                reps=2,
                entanglement="linear"
            )
        )
    ]

    all_results = []

    for name, feature_map in experiments:

        result = evaluate_feature_map(
            name,
            feature_map,
            X_train,
            X_test,
            y_train,
            y_test,
            results_dir
        )

        all_results.append(result)

    # Save comparison CSV
    csv_path = results_dir / "feature_map_comparison.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys()
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 50)
    print("FINAL FEATURE MAP COMPARISON")
    print("=" * 50)

    for result in all_results:

        print(
            result["Experiment"],
            "| Accuracy:", round(result["Accuracy"], 4),
            "| Balanced Accuracy:",
            round(result["Balanced Accuracy"], 4),
            "| ROC-AUC:", round(result["ROC-AUC"], 4)
        )

    print("\nComparison saved to:", csv_path)


if __name__ == "__main__":
    main()
