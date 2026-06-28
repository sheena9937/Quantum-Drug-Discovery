import pandas as pd
import matplotlib.pyplot as plt


def dataset_shape(df):
    print("\n========== Dataset Shape ==========")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")


def column_info(df):
    print("\n========== Column Information ==========")
    print(df.info())


def missing_values(df):
    print("\n========== Missing Values ==========")
    print(df.isnull().sum())


def class_distribution(df):
    print("\n========== Class Distribution ==========")

    counts = df["label"].value_counts()

    print(counts)

    plt.figure(figsize=(5,4))
    plt.bar(["Non-BBB", "BBB"], counts.sort_index())

    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Molecules")

    plt.tight_layout()
    plt.savefig("../../figures/class_distribution.png", dpi=300)
    plt.show()


def duplicate_smiles(df):
    duplicates = df["SMILES"].duplicated().sum()

    print("\n========== Duplicate Molecules ==========")
    print(f"Duplicate SMILES : {duplicates}")


def smiles_statistics(df):

    lengths = df["SMILES"].str.len()

    print("\n========== SMILES Length Statistics ==========")
    print(lengths.describe())


def main():

    df = pd.read_csv("../../data/raw/BBBP.csv")

    dataset_shape(df)

    column_info(df)

    missing_values(df)

    class_distribution(df)

    duplicate_smiles(df)

    smiles_statistics(df)


if __name__ == "__main__":
    main()