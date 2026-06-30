import numpy as np

from sklearn.model_selection import train_test_split


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