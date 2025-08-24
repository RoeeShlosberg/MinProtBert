import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Load Predictions and Labels
# -----------------------------------

def load_results(results_path):
    """Load predictions and labels from a CSV file."""
    df = pd.read_csv(results_path)
    predictions = df['predictions'].values
    labels = df['labels'].values
    return predictions, labels

# -----------------------------------
# 2. Calculate AUC
# -----------------------------------

def calculate_auc(predictions, labels):
    """Calculate ROC-AUC score."""
    return roc_auc_score(labels, predictions)

# -----------------------------------
# 3. Plot ROC Curve
# -----------------------------------

def plot_roc_curve(predictions, labels, output_path):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(labels, predictions)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(labels, predictions):.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

# -----------------------------------
# 4. Main Execution
# -----------------------------------

if __name__ == "__main__":
    RESULTS_PATH = "../savings/results/dna_binding/deepWET.csv"
    OUTPUT_PATH = "../savings/results/dna_binding/roc_curve.png"

    predictions, labels = load_results(RESULTS_PATH)
    auc = calculate_auc(predictions, labels)
    print(f"ROC-AUC Score: {auc:.4f}")

    plot_roc_curve(predictions, labels, OUTPUT_PATH)
