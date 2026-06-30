import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy(csv_path):

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6,5))

    plt.bar(df["Model"], df["Accuracy"])

    plt.title("Accuracy Comparison")

    plt.ylabel("Accuracy")

    plt.tight_layout()

    plt.savefig("../../figures/model_accuracy_comparison.png")

    plt.show()


def plot_auc(csv_path):

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6,5))

    plt.bar(df["Model"], df["ROC-AUC"])

    plt.title("ROC-AUC Comparison")

    plt.ylabel("ROC-AUC")

    plt.tight_layout()

    plt.savefig("../../figures/model_auc_comparison.png")

    plt.show()