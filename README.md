# MinProtBERT: Efficient Protein Language Model

## Overview
MinProtBERT is a compact and efficient version of ProteinBERT, designed to predict DNA-binding proteins while maintaining high performance with significantly reduced computational requirements.  
The project also evaluates the model's generalization capabilities on additional bioinformatics tasks, such as **secondary structure prediction (Q3)** and **membrane protein classification (Q2)**.  
This work demonstrates the potential of MinProtBERT as a practical solution for diverse bioinformatics applications.

The project builds upon **DistilProtBERT**, which highlighted the power of knowledge distillation in creating compact and efficient protein language models.  
MinProtBERT extends these ideas to further optimize both performance and computational efficiency.

The workflow of MinProtBERT involves three stages:
1. **Data Processing** – Preparing and filtering datasets, including generating subsets (cut and sub) to evaluate robustness and generalization.
2. **Pre-training** – Training MinProtBERT on large-scale protein sequence datasets to learn generalized sequence representations.
3. **Fine-tuning** – Adapting the pre-trained model to specific tasks: DNA-binding prediction, secondary structure prediction (Q3), and membrane protein classification (Q2).

---

## Authors
- Roee Shlosberg  
- Idan Korenfeld  
- **Supervisor:** Prof. Ron Unger  

---

## Summary
DNA-binding proteins are central to gene regulation, genome replication, and DNA repair. Computational prediction of these proteins is challenging due to the complex relationship between sequence, structure, and function.  
MinProtBERT leverages **Transformer-based Protein Language Models (PLMs)** to address this challenge efficiently.  

Using **Knowledge Distillation**, MinProtBERT compresses ProteinBERT into a smaller yet highly effective model.  
The training pipeline includes:
- **Pre-training** on large unlabeled protein sequence datasets (e.g., UniRef50).  
- **Fine-tuning** on labeled datasets for DNA-binding prediction, secondary structure prediction (Q3), and membrane protein classification (Q2).  

---

## Project Structure
```
MinProtBert/
├── data_processing/ # Scripts for dataset preparation (e.g., cut/sub processing, UniRef50 filtering)
├── datasets/ # All datasets used in pre-training and fine-tuning
├── finetunes/ # Fine-tuning scripts (dna_binding, Q2, Q3)
├── pretraining/ # Pre-training script for MinProtBERT
├── savings/ # Trained models, logs, and scores
├── MinProtBert_Report.pdf # Full written report (submitted document)
├── requirements.txt # Project dependencies
└── README.md
```

---

## Datasets
All datasets used in this project were **manually downloaded from public sources**, except for UniRef50 and the cut/sub datasets, which required additional preprocessing. Below is a detailed overview:

| Dataset                  | Description                                           | Location                              | Notes                          |
|--------------------------|-------------------------------------------------------|---------------------------------------|--------------------------------|
| **DNA-binding (original)** | Labeled protein sequences for DNA-binding classification | `datasets/original_dna_binding/`      | Manual download               |
| **DNA-binding (cut)**     | Randomly filtered subsets to reduce redundancy        | `datasets/cut_*/`                     | Generated via processing scripts |
| **DNA-binding (sub)**     | Similarity-based filtered subsets to test generalization | `datasets/sub_*/`                     | Generated via processing scripts |
| **Organism-specific DNA-binding** | DNA-binding proteins from specific organisms | `datasets/bacteria_dna_binding/`, `datasets/cElegans_dna_binding/` | Manual download |
| **Secondary structure (Q3)** | Helix / Strand / Coil labels                        | `datasets/secondary_structure/`        | Manual download               |
| **Membrane vs. soluble (Q2)** | Membrane vs. soluble proteins                     | `datasets/membrane_soulable/`          | Manual download               |
| **UniRef50**              | Large pretraining corpus, clustered at 50%           | `datasets/uniref50/`                  | Requires preprocessing         |
| **Deep-WET benchmark**    | External DNA-binding benchmark dataset               | `datasets/deepWet_dna_binding/`        | Manual download               |

For detailed instructions on obtaining and preparing the datasets, refer to the **[MinProtBert Report](./MinProtBert_Report.pdf)**.

---

## DNA-Binding Classification
The DNA-binding classification task involves multiple runs to evaluate the model's performance under different conditions:

- **Full Dataset**: Training and testing on the complete dataset.
- **Filtered Subsets**: Using `cut` (random filtering) and `sub` (similarity-based filtering) subsets to test the model's ability to generalize and avoid overfitting.
- **Cross-Organism Evaluation**: Testing the model on sequences from different organisms (e.g., humans, bacteria) to assess its robustness across diverse data.

The fine-tuning process for DNA-binding involves:
1. Training the model on the DNA-binding dataset.
2. Evaluating performance using metrics such as Accuracy, AUC, and MCC.
3. Comparing results with baseline models like Deep-WET to highlight MinProtBERT's efficiency and accuracy.

---

## Usage
This repository documents the full workflow carried out during the project.  
Scripts for pre-training and fine-tuning are included for completeness, but re-running them requires access to the processed datasets listed above.  

### Pre-training
```bash
python pretraining/pre_training.py
```

### Fine-tuning
- **DNA-binding**:
  ```bash
  python finetunes/dna_binding/finetune_dna_binding.py
  ```
- **Secondary structure prediction (Q3)**:
  ```bash
  python finetunes/q3/finetune_q3.py
  ```
- **Membrane protein classification (Q2)**:
  ```bash
  python finetunes/q2/finetune_q2.py
  ```

Note: These scripts were used in the experiments documented in the **[Report](./MinProtBert_Report.pdf)** and are provided for reference only.

---

## Results
MinProtBERT achieves nearly identical performance to ProteinBERT across all evaluated tasks, while significantly reducing runtime:

- **DNA-binding prediction**: ~38% runtime reduction.
- **Secondary structure prediction (Q3)**: ~66% runtime reduction.
- **Membrane protein classification (Q2)**: ~16% runtime reduction, with improved accuracy.

For detailed results and discussion, see the **[Project Report](./MinProtBert_Report.pdf)**.

---

## Bibliography
1. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
2. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.
3. Brandes, N., et al. (2022). ProteinBERT: A universal deep-learning model of protein sequence and function. Bioinformatics.
4. Sanh, V., et al. (2019). DistilBERT: A distilled version of BERT. arXiv.
5. Geffen, Y., et al. (2022). DistilProtBert: Distillation of Protein Language Models. bioRxiv.

---

## Citation
If you use MinProtBERT report and its insights in your research, please cite this repository and refer to the **[MinProtBert_Report.docx](./MinProtBert_Report.pdf)** for methodology and results.