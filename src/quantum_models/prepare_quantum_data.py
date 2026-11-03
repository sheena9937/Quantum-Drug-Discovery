
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def main():

    # Automatically locate project root
    project_root = Path(__file__).resolve().parents[2]

    data_dir = project_root / "data" / "processed"

    fingerprint_path = data_dir / "fingerprints.npy"
    labels_path = data_dir / "labels.npy"

    # Check whether input files exist
    if not fingerprint_path.exists():
        raise FileNotFoundError(
            f"Fingerprint file not found: {fingerprint_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}"
        )

    # Load molecular fingerprints and labels
    X = np.load(fingerprint_path)
    y = np.load(labels_path)

    print("Original feature shape:", X.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # Reduce features from 1024 to 4
    pca = PCA(n_components=4, random_state=42)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("\nAfter PCA:")
    print("Training shape:", X_train_pca.shape)
    print("Testing shape:", X_test_pca.shape)

    print(
        "Explained variance ratio:",
        pca.explained_variance_ratio_
    )

    # Scale features
    scaler = MinMaxScaler()

    X_train_quantum = scaler.fit_transform(X_train_pca)
    X_test_quantum = scaler.transform(X_test_pca)

    # Save processed quantum data
    np.save(data_dir / "quantum_X_train.npy", X_train_quantum)
    np.save(data_dir / "quantum_X_test.npy", X_test_quantum)
    np.save(data_dir / "quantum_y_train.npy", y_train)
    np.save(data_dir / "quantum_y_test.npy", y_test)

    print("\nQuantum data preparation completed.")
    print("Saved files in:", data_dir)


if __name__ == "__main__":
    main()
