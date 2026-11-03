
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


def main():

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed"
    results_dir = project_root / "results" / "comparisons"

    results_dir.mkdir(parents=True, exist_ok=True)

    # Load prepared quantum data
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
    feature_map = z_feature_map(
        feature_dimension=4,
        reps=1
    )

    # Create trainable quantum circuit
    ansatz = real_amplitudes(
        num_qubits=4,
        reps=1
    )

    print("\nFeature Map:")
    print(feature_map)

    print("\nTrainable Ansatz:")
    print(ansatz)

    print("\nNumber of trainable parameters:", ansatz.num_parameters)

    # Track optimization progress
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

    # Classical optimizer
    optimizer = COBYLA(maxiter=30)

    # Reproducible initial parameters
    rng = np.random.default_rng(42)

    initial_point = rng.uniform(
        -np.pi,
        np.pi,
        ansatz.num_parameters
    )

    # Create Variational Quantum Classifier
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        loss="cross_entropy",
        initial_point=initial_point,
        callback=callback
    )

    print("\nTraining VQC...")

    vqc.fit(X_train_small, y_train_small)

    print("\nVQC training completed!")

    # Predict test labels
    y_pred = vqc.predict(X_test_small)

    # Get predicted probabilities
    y_prob = vqc.predict_proba(X_test_small)

    # Evaluation
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

    # Save predictions
    np.save(
        results_dir / "vqc_predictions.npy",
        y_pred
    )

    # Save predicted probabilities
    np.save(
        results_dir / "vqc_probabilities.npy",
        y_prob
    )

    # Save trained parameters
    np.save(
        results_dir / "vqc_weights.npy",
        vqc.weights
    )

    # Save optimization history
    np.save(
        results_dir / "vqc_loss_history.npy",
        np.array(loss_history)
    )

    # Save metrics
    np.savez(
        results_dir / "vqc_metrics.npz",
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        roc_auc=auc,
        confusion_matrix=cm,
        train_samples=len(y_train_small),
        test_samples=len(y_test_small)
    )

    print("\nVQC results saved successfully.")


if __name__ == "__main__":
    main()
