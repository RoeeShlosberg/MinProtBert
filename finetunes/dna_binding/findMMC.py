import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef

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
# 2. Calculate MCC
# -----------------------------------

def calculate_mcc(predictions, labels):
    """Calculate Matthews Correlation Coefficient (MCC)."""
    return matthews_corrcoef(labels, predictions)

# -----------------------------------
# 3. Main Execution
# -----------------------------------

if __name__ == "__main__":
    RESULTS_PATH = "../savings/results/dna_binding/deepWET.csv"

    predictions, labels = load_results(RESULTS_PATH)
    mcc = calculate_mcc(predictions, labels)
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
