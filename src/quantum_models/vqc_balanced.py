
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from qiskit.circuit.library import z_feature_map, real_amplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC


def oversample_minority_class(X, y, random_state=42):
    rng = np.random.default_rng(random_state)

    classes, counts = np.unique(y, return_counts=True)
    target_count = counts.max()

    X_parts = []
    y_parts = []

    for class_label, count in zip(classes, counts):
        class_indices = np.where(y == class_label)[0]

        if count < target_count:
            extra_indices = rng.choice(
                class_indices,
                size=target_count - count,
                replace=True
            )
            class_indices = np.concatenate(
                [class_indices, extra_indices]
            )

        X_parts.append(X[class_indices])
        y_parts.append(y[class_indices])

    X_balanced = np.concatenate(X_parts)
    y_balanced = np.concatenate(y_parts)

    permutation = rng.permutation(len(y_balanced))

    return X_balanced[permutation], y_balanced[permutation]


def main():
    project_root = Path(__file__).resolve().parents[2]

    data_dir = project_root / "data" / "processed"
    results_dir = project_root / "results" / "comparisons"

    results_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(data_dir / "quantum_X_train.npy")
    X_test = np.load(data_dir / "quantum_X_test.npy")
    y_train = np.load(data_dir / "quantum_y_train.npy")
    y_test = np.load(data_dir / "quantum_y_test.npy")

    X_train_small, _, y_train_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=120,
        random_state=42,
        stratify=y_train
    )

    X_test_small, _, y_test_small, _ = train_test_split(
        X_test,
        y_test,
        train_size=100,
        random_state=42,
        stratify=y_test
    )

    print("Original training distribution:")
    print(np.unique(y_train_small, return_counts=True))

    X_balanced, y_balanced = oversample_minority_class(
        X_train_small,
        y_train_small
    )

    print("\nBalanced training distribution:")
    print(np.unique(y_balanced, return_counts=True))

    feature_map = z_feature_map(
        feature_dimension=4,
        reps=1
    )

    ansatz = real_amplitudes(
        num_qubits=4,
        reps=1
    )

    loss_history = []

    def callback(weights, objective_value):
        loss_history.append(objective_value)

        if len(loss_history) % 5 == 0:
            print(
                "Iteration:",
                len(loss_history),
                "Loss:",
                objective_value
            )

    optimizer = COBYLA(maxiter=20)

    rng = np.random.default_rng(42)

    initial_point = rng.uniform(
        -np.pi,
        np.pi,
        ansatz.num_parameters
    )

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        loss="cross_entropy",
        initial_point=initial_point,
        callback=callback
    )

    print("\nTraining balanced VQC...")

    vqc.fit(X_balanced, y_balanced)

    print("\nTraining completed!")

    y_pred = vqc.predict(X_test_small)
    y_prob = vqc.predict_proba(X_test_small)

    accuracy = accuracy_score(y_test_small, y_pred)

    balanced_acc = balanced_accuracy_score(
        y_test_small,
        y_pred
    )

    auc = roc_auc_score(
        y_test_small,
        y_prob[:, 1]
    )

    cm = confusion_matrix(
        y_test_small,
        y_pred,
        labels=[0, 1]
    )

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
            labels=[0, 1],
            zero_division=0
        )
    )

    np.save(
        results_dir / "vqc_balanced_predictions.npy",
        y_pred
    )

    np.save(
        results_dir / "vqc_balanced_probabilities.npy",
        y_prob
    )

    np.save(
        results_dir / "vqc_balanced_weights.npy",
        vqc.weights
    )

    np.save(
        results_dir / "vqc_balanced_loss_history.npy",
        np.array(loss_history)
    )

    np.savez(
        results_dir / "vqc_balanced_metrics.npz",
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        roc_auc=auc,
        confusion_matrix=cm,
        train_samples=len(y_balanced),
        test_samples=len(y_test_small)
    )

    print("\nBalanced VQC results saved successfully.")


if __name__ == "__main__":
    main()
