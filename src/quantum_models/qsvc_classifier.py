
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
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

    print("Original training shape:", X_train.shape)
    print("Original testing shape:", X_test.shape)

    # Select a manageable stratified training subset
    X_train_small, _, y_train_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=120,
        random_state=42,
        stratify=y_train
    )

    # Select a manageable stratified test subset
    X_test_small, _, y_test_small, _ = train_test_split(
        X_test,
        y_test,
        train_size=100,
        random_state=42,
        stratify=y_test
    )

    print("\nPilot training samples:", len(y_train_small))
    print("Pilot testing samples:", len(y_test_small))

    # Create quantum feature map
    feature_map = zz_feature_map(
        feature_dimension=4,
        reps=1,
        entanglement="linear"
    )

    print("\nQuantum Feature Map:")
    print(feature_map)

    # Create quantum kernel
    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    # Create quantum SVM
    qsvc = QSVC(
        quantum_kernel=quantum_kernel
    )

    print("\nTraining QSVC...")

    qsvc.fit(X_train_small, y_train_small)

    print("Training completed!")

    # Predictions
    y_pred = qsvc.predict(X_test_small)

    # Evaluation
    accuracy = accuracy_score(y_test_small, y_pred)
    cm = confusion_matrix(y_test_small, y_pred)

    print("\nAccuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test_small, y_pred))

    # ROC-AUC
    decision_scores = qsvc.decision_function(X_test_small)
    auc = roc_auc_score(y_test_small, decision_scores)

    print("ROC-AUC:", auc)

    # Save results
    results_dir = project_root / "results" / "comparisons"
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "qsvc_predictions.npy", y_pred)

    np.savez(
        results_dir / "qsvc_metrics.npz",
        accuracy=accuracy,
        roc_auc=auc,
        confusion_matrix=cm,
        train_samples=len(y_train_small),
        test_samples=len(y_test_small)
    )

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
