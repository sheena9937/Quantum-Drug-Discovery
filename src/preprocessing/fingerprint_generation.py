import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np


def load_dataset(path):
    df = pd.read_csv(path)
    return df


def smiles_to_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol

def generate_fingerprint(molecule, radius=2, n_bits=1024):

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits
    )

    fingerprint = generator.GetFingerprint(molecule)

    array = np.zeros((n_bits,), dtype=int)

    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return array

def create_feature_matrix(smiles_list):

    fingerprints = []

    total = len(smiles_list)

    for index, smiles in enumerate(smiles_list):

        molecule = smiles_to_molecule(smiles)

        if molecule is None:
            print(f"Invalid molecule at index {index}")
            continue

        fingerprint = generate_fingerprint(molecule)

        fingerprints.append(fingerprint)

        if (index + 1) % 250 == 0 or index == total - 1:
            print(f"Processed {index + 1}/{total} molecules")

    return np.array(fingerprints)

def save_features(features, labels):

    np.save("../../data/processed/fingerprints.npy", features)
    np.save("../../data/processed/labels.npy", labels)

    print("\nFeatures saved successfully!")


def main():

    dataset = load_dataset("../../data/raw/BBBP.csv")

    print("Dataset loaded successfully!")
    print()

    smiles = dataset["SMILES"]

    X = create_feature_matrix(smiles)
    y = dataset["label"].to_numpy()

    save_features(X, y)

    print("\nFeature Matrix Shape:")
    print(X.shape)

    print("\nFirst Molecule (First 20 Bits):")
    print(X[0][:20])

if __name__ == "__main__":
    main()