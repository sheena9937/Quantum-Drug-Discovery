import pandas as pd


def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)

        print("Dataset loaded successfully!")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        return df

    except FileNotFoundError:
        print("Dataset not found.")
        return None

    except Exception as e:
        print("Something went wrong:", e)
        return None


if __name__ == "__main__":
    dataset_path = "../../data/raw/BBBP.csv"

    df = load_dataset(dataset_path)

    if df is not None:
        print("\nFirst 5 rows:\n")
        print(df.head())