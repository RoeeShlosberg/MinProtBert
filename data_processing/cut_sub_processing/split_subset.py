import os
import pandas as pd

# Path to the dataset and target directory for splits
DATA = "../datasets/sub_0.15_dna_binding/combined_subset_0.15.csv"
TARGET = "../datasets/sub_0.15_dna_binding/splits"

# Proportion of data to allocate for training
TRAIN_ODD = 0.85

# Load the dataset from CSV
df = pd.read_csv(DATA)

# Create the target directory if it does not exist
if not os.path.exists(TARGET):
    os.makedirs(TARGET)

# Shuffle the data randomly for unbiased splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into training and remaining (val+test) sets
train_size = int(len(df) * TRAIN_ODD)
train_df = df[:train_size]
val_df = df[train_size:]

# Further split the remaining data equally into validation and test sets
val_size = len(val_df) // 2
val_df, test_df = val_df[:val_size], val_df[val_size:]

# Save the splits into separate CSV files
train_df.to_csv(os.path.join(TARGET, "train.csv"), index=False)
val_df.to_csv(os.path.join(TARGET, "val.csv"), index=False)
test_df.to_csv(os.path.join(TARGET, "test.csv"), index=False)

# Print out the sizes of each dataset split
print(f"Data split into train ({len(train_df)}), validation ({len(val_df)}), and test ({len(test_df)}) sets.")
