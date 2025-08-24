# MinProtBERT: Efficient Protein Language Model

## Overview
MinProtBERT is a compact and efficient version of ProteinBERT, designed to predict DNA-binding proteins while maintaining high performance with significantly reduced computational requirements. The project also evaluates the model's generalization capabilities on additional bioinformatics tasks, such as secondary structure prediction (Q3) and membrane protein classification (Q2). MinProtBERT demonstrates its potential as a practical solution for diverse bioinformatics applications.

This work was inspired by **DistilProtBERT**, which demonstrated the power of knowledge distillation in creating compact and efficient protein language models. MinProtBERT builds upon these ideas to further optimize performance and computational efficiency.

The workflow of MinProtBERT involves three main stages:
1. **Data Processing**: Preparing and filtering datasets for training and evaluation. This includes generating subsets (e.g., `cut` and `sub`) to test the model's generalization capabilities and reduce redundancy in the data.
2. **Pre-training**: Training the MinProtBERT model on large-scale protein sequence datasets to learn generalized protein representations.
3. **Fine-tuning**: Adapting the pre-trained model to specific tasks, such as DNA-binding classification, secondary structure prediction, and membrane protein classification.

## Authors
- **Roee Shlosberg**
- **Idan Korenfeld**
- **Supervisor**: Prof. Ron Unger

## Summary
DNA-binding proteins play critical roles in gene regulation, genome replication, and DNA repair. Predicting DNA-binding proteins computationally is challenging due to the complex relationship between sequence, structure, and function. MinProtBERT leverages the power of Protein Language Models (PLMs) based on Transformers to address this challenge efficiently.

The project employs Knowledge Distillation to create MinProtBERT, a distilled version of ProteinBERT. The training process includes:
1. **Pre-training**: Learning generalized protein sequence representations.
2. **Fine-tuning**: Adapting the model for specific classification tasks, including DNA-binding prediction, secondary structure prediction, and membrane protein classification.

## Project Structure
```
MinProtBert/
├── data_processing/
├── datasets/
├── finetunes/
├── pretraining/
├── savings/
├── README.md
├── requirements.txt
└── דוח סיכום.docx
```
- **data_processing/**: Scripts for dataset preparation and filtering. For example, `cut` and `sub` processing scripts create subsets of data to evaluate the model's robustness.
- **datasets/**: Contains datasets for training and evaluation, such as DNA-binding proteins, secondary structure data, and membrane protein data.
- **finetunes/**: Fine-tuning scripts for specific tasks. Each task (e.g., DNA-binding, Q2, Q3) has its own directory with scripts for training and evaluation.
- **pretraining/**: Scripts for pre-training the MinProtBERT model on large protein sequence datasets.
- **savings/**: Stores logs, results, and trained models.

## DNA-Binding Classification
The DNA-binding classification task involves multiple runs to evaluate the model's performance under different conditions:
- **Full Dataset**: Training and testing on the complete dataset.
- **Filtered Subsets**: Using `cut` (random filtering) and `sub` (similarity-based filtering) subsets to test the model's ability to generalize and avoid overfitting.
- **Cross-Organism Evaluation**: Testing the model on sequences from different organisms (e.g., humans, bacteria) to assess its robustness across diverse data.

The fine-tuning process for DNA-binding involves:
1. Training the model on the DNA-binding dataset.
2. Evaluating performance using metrics such as Accuracy, AUC, and MCC.
3. Comparing results with baseline models like Deep-WET to highlight MinProtBERT's efficiency and accuracy.

## Usage
### Pre-training
Run the pre-training script to initialize the MinProtBERT model:
```bash
python pretraining/pre_training.py
```

### Fine-tuning
Fine-tune the model for specific tasks:
- DNA-binding:
  ```bash
  python finetunes/dna_binding/finetune_dna_binding.py
  ```
- Secondary structure prediction (Q3):
  ```bash
  python finetunes/q3/finetune_q3.py
  ```
- Membrane protein classification (Q2):
  ```bash
  python finetunes/q2/finetune_q2.py
  ```

### Evaluation
Evaluate the fine-tuned models using the provided scripts in each task directory.

## Results
MinProtBERT achieves near-identical performance to ProteinBERT across all tasks while significantly reducing runtime:
- **DNA-binding prediction**: 38% runtime reduction.
- **Secondary structure prediction**: 66% runtime reduction.
- **Membrane protein classification**: 16% runtime reduction with improved accuracy.

## Bibliography
1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.
3. Brandes, N., Ofer, D., Peleg, Y., Rappoport, N., & Linial, M. (2022). ProteinBERT: A universal deep-learning model of protein sequence and function. Bioinformatics.
4. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.
5. Geffen, Y., Ofran, Y., & Unger, R. (2022). DistilProtBert: Distillation of Protein Language Models. bioRxiv.

## Citation
If you use MinProtBERT in your research, please cite the project and refer to the [Summary Report](./דוח%20סיכום.docx) for detailed methodology and results.