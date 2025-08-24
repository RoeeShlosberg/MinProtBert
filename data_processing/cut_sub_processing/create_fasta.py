import pandas as pd

PATH = "../datasets/original_dna_binding/combined_binding_data.csv"

# create fasta file
with open("../datasets/original_dna_binding/combined_data.fasta", "w") as f:
    data = pd.read_csv(PATH)
    for i, row in data.iterrows():
        f.write(f">{i}\n{row['Sequence']}\n")
