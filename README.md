# PDI-Peptide Interaction Prediction

This repository contains a machine learning pipeline designed to identify and predict the interaction between peptides and Protein Disulfide Isomerase (PDI). By leveraging ESM (Evolutionary Scale Modeling) sequence embeddings and a Logistic Regression classifier, this tool predicts binding affinity and estimates potential physical contacts.

## 📂 Repository Structure

* **`protein_peptide_contacts.csv`**: Original training dataset containing sequences and experimental contact data.
* **`Pdi_peptides.csv`**: Target dataset containing the novel peptide sequences for PDI interaction prediction.
* **`predict_2.py`**: Python script to execute predictions on new peptide sequences using the trained model.
* **`prediccion_pdi_peptidos.ipynb`**: Interactive Google Colab notebook containing the complete pipeline (data cleaning, ESM embedding generation, model training, and cross-validation).

## 🚀 Usage

The most straightforward way to execute this pipeline is through Google Colab. The provided notebook is configured to automatically download the required datasets and Python scripts directly from this repository.

To run the prediction script locally, ensure you have the required dependencies installed and use the following command:

Bash
python predict_2.py --clf-path ppi_esm_logreg_final_model.pkl --new-data Pdi_peptides.csv --out ppi_prediction_final_esm_new.csv


## 👩‍🔬 Authors

* **PhD(c) Francisca Rodríguez Cabello** - Doctoral Biomedical Sciences student UTALCA, Chile
* **Abigail Elisabeth Teitgen** - Postdoctoral MSCA Researcher, Spain
* **María Eugenia Ulzurrun de Asanza Vega** - CSIC Researcher, Spain
* **Nuria Eugenia Campillo Martin** - Group Leader Computational Intelligence for Drug Discovery in Biomedicine CSIC, Spain


📄 License
This project is open-source and available under the MIT License.
