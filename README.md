# UNO

The UNO model consists of two fully connected neural network branches for separately embedding the drug and cell line features, then a third fully connected neural network to regress on the combined embedded features. UNO was one of the benchmark models in the CANDLE project for cancer drug response.

This repository uses the [IMPROVE API](https://jdacs4c-improve.github.io/docs/content/API.html) for building this drug response prediction (DRP) model. It uses the[cross-study analysis (CSA) dataset](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

The `workflows` folder provides examples for `cross study analysis`, `learning curve analysis` and `deephyper hyperparameter optimization`

## Dependencies
Installation instuctions are detailed below in [Step-by-step instructions](#step-by-step-instructions).

ML framework:
+ [TensorFlow](https://www.tensorflow.org/) -- deep learning framework for building the prediction model

IMPROVE dependencies:
+ [IMPROVE Library](https://github.com/JDACS4C-IMPROVE/IMPROVE)


## Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

The data tree is shown below:
```
csa_data/raw_data/
├── splits
│   ├── CCLE_all.txt
│   ├── CCLE_split_0_test.txt
│   ├── CCLE_split_0_train.txt
│   ├── CCLE_split_0_val.txt
│   ├── CCLE_split_1_test.txt
│   ├── CCLE_split_1_train.txt
│   ├── CCLE_split_1_val.txt
│   ├── ...
│   ├── GDSCv2_split_9_test.txt
│   ├── GDSCv2_split_9_train.txt
│   └── GDSCv2_split_9_val.txt
├── x_data
│   ├── cancer_copy_number.tsv
│   ├── cancer_discretized_copy_number.tsv
│   ├── cancer_DNA_methylation.tsv
│   ├── cancer_gene_expression.tsv
│   ├── cancer_miRNA_expression.tsv
│   ├── cancer_mutation_count.tsv
│   ├── cancer_mutation_long_format.tsv
│   ├── cancer_mutation.parquet
│   ├── cancer_RPPA.tsv
│   ├── drug_ecfp4_nbits512.tsv
│   ├── drug_info.tsv
│   ├── drug_mordred_descriptor.tsv
│   └── drug_SMILES.tsv
└── y_data
    └── response.tsv
```

## Model scripts and parameter file
+ `uno_preprocess_improve.py` - takes benchmark data files and transforms them into files for training and inference
+ `uno_train_improve.py` - trains the UNO model
+ `uno_infer_improve.py` - runs inference with the trained UNO model
+ `uno_params.ini` - default parameter file (parameter values specified in this file override the defaults)
+ `model_params_def.py` - definitions of parameters that are specific to the model

# Step-by-step instructions

### 1. Clone the model repository
```
git clone https://github.com/JDACS4C-IMPROVE/UNO
cd UNO
git checkout develop
```


### 2. Set computational environment
Create conda environment using the provided environment file:
```bash
conda env create -f uno_environment.yml
conda activate uno_env
```

### 3. Run `setup_improve.sh`.
```bash
source setup_improve.sh
```

This will:
1. Download cross-study analysis (CSA) benchmark data into `./csa_data/`.
2. Clone IMPROVE repo (checkout `develop`) outside the UNO model repo.
3. Set `PYTHONPATH` (adds IMPROVE repo).
4. Set `IMPROVE_DATA_DIR`.
5. Note that you must run this to setup the path variables every time you log in.  Installation is skipped if the directories already exist.

Alternatively, just set the IMPROVE_DATA_DIR after step 2. The new updates to conda_env.yml already installs `improvelib`.


### 4. Preprocess CSA benchmark data (_raw data_) to construct model input data (_ML data_)
```bash
python uno_preprocess_improve.py --input_dir ./csa_data/raw_data --output_dir exp_result
```

Preprocesses the CSA data and creates train, validation (val), and test datasets.

Generates:
* nine model input data files (each has a file for train, val, and infer): `ge_*_data.parquet`, `md_*_data.parquet`, `rsp_*_data.parquet`
* three tabular data files, each containing the drug response values (i.e. AUC) and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

```
exp_result
├── ge_test_data.parquet
├── ge_train_data.parquet
├── ge_val_data.parquet
├── md_test_data.parquet
├── md_train_data.parquet
├── md_val_data.parquet
├── param_log_file.txt
├── rsp_test_data.parquet
├── rsp_train_data.parquet
├── rsp_val_data.parquet
├── test_y_data.csv
├── train_y_data.csv
├── val_y_data.csv
├── x_data_gene_expression_scaler.gz
└── x_data_mordred_scaler.gz
```

### 5. Train UNO model
```bash
python uno_train_improve.py --input_dir exp_result --output_dir exp_result
```

Trains UNO using the model input data: `ge_train_data.parquet`, `md_train_data.parquet`, `rsp_train_data.parquet` (training) and `ge_val_data.parquet`, `md_val_data.parquet`, `rsp_val_data.parquet` (for early stopping).

Generates:
* trained model: `saved_model.pb`
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`

```
exp_result
├── ge_test_data.parquet
├── ge_train_data.parquet
├── ge_val_data.parquet
├── md_test_data.parquet
├── md_train_data.parquet
├── md_val_data.parquet
├── model
    ├── assets/
    ├── keras_metadata.pb
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
├── param_log_file.txt
├── rsp_test_data.parquet
├── rsp_train_data.parquet
├── rsp_val_data.parquet
├── test_y_data.csv
├── train_y_data.csv
├── val_scores.json
├── val_y_data.csv
├── val_y_data_predicted.csv
├── x_data_gene_expression_scaler.gz
└── x_data_mordred_scaler.gz
```

### 6. Run inference on test data with the trained model
```bash
python uno_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
```
exp_result
├── ge_test_data.parquet
├── ge_train_data.parquet
├── ge_val_data.parquet
├── md_test_data.parquet
├── md_train_data.parquet
├── md_val_data.parquet
├── model
    ├── assets/
    ├── keras_metadata.pb
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
├── param_log_file.txt
├── rsp_test_data.parquet
├── rsp_train_data.parquet
├── rsp_val_data.parquet
├── test_scores.json
├── test_y_data.csv
├── test_y_data_predicted.csv
├── train_y_data.csv
├── val_scores.json
├── val_y_data.csv
├── val_y_data_predicted.csv
├── x_data_gene_expression_scaler.gz
└── x_data_mordred_scaler.gz
```
