
# UniDBind

**Unified sequence-based prediction of DNA-binding across structured and disordered proteins**

## Introduction

This is the official repository of UniDBind, a sequence-based computational method for the unified prediction of DNA-binding proteins, covering both structured proteins and intrinsically disordered proteins (IDPs).

## Environment Requirements

All software dependencies and runtime environments are listed in the YAML configuration file. You can build the conda environment with the following command:

```bash
conda env create -f environment.yml
```

## Datasets

Detailed dataset information, including raw data, data processing guidelines, and benchmark collections, are available on our official webserver:

[http://bliulab.net/UniDBind/](http://bliulab.net/UniDBind/)

## Model Usage

All training and evaluation scripts are placed in the `./scripts` folder.

### 1. Model Training

To train the UniDBind model from scratch, please execute the training script:

```bash
bash scripts/run_train.sh
```

### 2. Model Evaluation

To evaluate the trained model on test datasets, run the evaluation script as follows:

```bash
bash scripts/run_evaluate_hybrid.sh
```

## Feature Preparation

UniDBind utilizes multiple sequence-derived features for model prediction, including **PSSM, physicochemical , and ESM2** embedding features. All feature extraction procedures are integrated into the project pipeline.

## Contact

If you have any questions or suggestions, please feel free to contact us or raise an issue in this repository.
