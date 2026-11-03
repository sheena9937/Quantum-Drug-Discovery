
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


def main():

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed"

    # Load quantum-ready data
    X_train = np.load(data_dir / "quantum_X_train.npy")
    X_test = np.load(data_dir / "quantum_X_test.npy")
    y_train = np.load(data_dir / "quantum_y_train.npy")
    y_test = np.load(data_dir / "quantum_y_test.npy")

    # Select the same pilot training subset
    X_train_small, _, y_train_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=120,
        random_state=42,
        stratify=y_train
    )

    # Select the same pilot testing subset
    X_test_small, _, y_test_small, _ = train_test_split(
        X_test,
        y_test,
        train_size=100,
        random_state=42,
        stratify=y_test
    )

    print("Training samples:", len(y_train_small))
    print("Testing samples:", len(y_test_small))

    print("\nTraining class distribution:")
    print(np.unique(y_train_small, return_counts=True))

    # Create quantum feature map
    feature_map = zz_feature_map(
        feature_dimension=4,
        reps=1,
        entanglement="linear"
    )

    # Create quantum kernel
    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    # Balanced QSVC
    qsvc = QSVC(
        quantum_kernel=quantum_kernel,
        class_weight="balanced"
    )

    print("\nTraining Balanced QSVC...")

    qsvc.fit(X_train_small, y_train_small)

    print("Training completed!")

    # Predictions
    y_pred = qsvc.predict(X_test_small)

    # Evaluation
    accuracy = accuracy_score(y_test_small, y_pred)
    balanced_acc = balanced_accuracy_score(y_test_small, y_pred)
    cm = confusion_matrix(y_test_small, y_pred)

    decision_scores = qsvc.decision_function(X_test_small)
    auc = roc_auc_score(y_test_small, decision_scores)

    print("\nAccuracy:", accuracy)
    print("Balanced Accuracy:", balanced_acc)
    print("ROC-AUC:", auc)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test_small,
            y_pred,
            zero_division=0
        )
    )

    # Save results
    results_dir = project_root / "results" / "comparisons"
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        results_dir / "qsvc_balanced_predictions.npy",
        y_pred
    )

    np.savez(
        results_dir / "qsvc_balanced_metrics.npz",
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        roc_auc=auc,
        confusion_matrix=cm,
        train_samples=len(y_train_small),
        test_samples=len(y_test_small)
    )

    print("\nBalanced QSVC results saved successfully.")


if __name__ == "__main__":
    main()

