import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_feature_importance(importance):

    plt.figure(figsize=(12,6))

    plt.bar(range(len(importance)), importance)

    plt.title("Feature Importance")

    plt.xlabel("Fingerprint Bit")

    plt.ylabel("Importance")

    plt.tight_layout()

    plt.savefig("../../figures/random_forest_feature_importance.png")

    plt.show()

def plot_top_features(importance):

    top_n = 20

    indices = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10,6))

    plt.barh(range(top_n), importance[indices])

    plt.yticks(range(top_n), indices)

    plt.xlabel("Feature Importance")

    plt.ylabel("Fingerprint Bit")

    plt.title("Top 20 Most Important Fingerprint Bits")

    plt.tight_layout()

    plt.savefig("../../figures/top20_feature_importance.png")

    plt.show()

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

