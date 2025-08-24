import pandas as pd
import random
import os

# --- Configuration ---
PATH = "../datasets/original_dna_binding/combined_binding_data.csv" 
MAX = 7041           # Max number of sequences to use (after shuffling)
TRAIN_ODD = 0.85     # Proportion of data to use for training

# --- Load and shuffle the dataset ---
data = pd.read_csv(PATH)                                
suffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the data deterministically
cut_data = suffled_data[:MAX]                           # Truncate to MAX examples

# --- Split into train, validation, and test sets ---
train_size = int(len(cut_data) * TRAIN_ODD)
train_data = cut_data[:train_size]                      
val_data = cut_data[train_size:]                        
val_size = len(val_data) // 2
val_data, test_data = val_data[:val_size], val_data[val_size:]  

# --- Prepare output directory ---
target_path = "../datasets/cut_0.15_dna_binding/splits"
if not os.path.exists(target_path):
    os.makedirs(target_path)

# --- Save splits to CSV files ---
train_data.to_csv(os.path.join(target_path, "train.csv"), index=False)
val_data.to_csv(os.path.join(target_path, "val.csv"), index=False)
test_data.to_csv(os.path.join(target_path, "test.csv"), index=False)

# --- Report summary ---
print(f"Data split into train ({len(train_data)}), validation ({len(val_data)}), and test ({len(test_data)}) sets.")
