import pandas as pd
from Bio import SeqIO

# Define file paths
FASTA_FILE = "../datasets/sub_0.15_dna_binding/subset_0.15.fasta"
OG_CSV = "../datasets/original_dna_binding/combined_binding_data.csv"

# Load the original CSV file into a DataFrame
df = pd.read_csv(OG_CSV)

# Normalize the sequence strings: remove whitespace and convert to uppercase
df['Sequence'] = df['Sequence'].str.strip().str.upper()

# Parse sequences from the FASTA file and store them in a set 
subset_seqs = set(
    str(record.seq).strip().upper() for record in SeqIO.parse(FASTA_FILE, "fasta")
)

# Filter the DataFrame: keep only rows with sequences found in the FASTA subset
filtered_df = df[df['Sequence'].isin(subset_seqs)]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv("../datasets/sub_0.15_dna_binding/combined_subset_0.15.csv", index=False)

# Print how many sequences were retained after filtering
print(f"Filtered dataset contains {len(filtered_df)} sequences matching the subset from {FASTA_FILE}.")
