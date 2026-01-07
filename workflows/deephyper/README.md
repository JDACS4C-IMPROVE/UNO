# README for Deephyper IMPROVE Workflow

This directory contains files necessary for running hyperparameter optimization using Deephyper in the IMPROVE project. Below is a brief description of each file:

## Files Description

1. **debug_small_submit.sh**: 
   - A shell script used to submit jobs to the ALCF Polaris computing cluster. This script is configured to run a small debug run on two nodes and not much variation in hyperparameters. Note: For running on lambda or other local machines with no submission scripts, follow the README in IMPROVE/workflow/deephyper.

2. **hpo_deephyper_hyperparameters.json**: 
   - A JSON file that contains the hyperparameters for the Deephyper optimization process. This file defines the search space for the hyperparameters. MODIFICATIONS are needed as per requirements.

3. **hpo_deephyper_params.ini**: 
   - An INI configuration file that specifies various parameters for the hyperparameter optimization process, including directories, model settings, and evaluation parameters. Note: Follow instructions on the IMPROVE/workflow/deephyper/ page.

4. **hpo_deephyper_params_def.py**: 
   - A Python script that defines default parameters and settings for the hyperparameter optimization process. This script may include functions or classes to facilitate the optimization workflow. Note, UNO workflow disables Singularity as this directly runs the workflow without containers.

## Instructions

1. **Move Files**: 
   Please move the above files to the following directory:
   ```
   <PATH>/IMPROVE/workflows/deephyper_hpo/
   ```

2. **Run the Submit Script**: 
   After moving the files, run the submit script on ALCF Polaris using the following command:
   ```bash
   qsub submit.sh
   ```

   Note: This is a small testing script, prod queue and adjustment in the above mentioned files are needed for larger meaningful runs

3. **Environment Requirements**: 
   Ensure that you have the required Conda environment set up. The environment should have both `mpi4py` and `deephyper` already installed. Additionally, the `Uno_IMPROVE` environment is required as per the hpo_deephyper_params.ini, see line  model_environment = Uno_IMPROVE; Also follow the steps here before launching submit.sh
   ```
   <PATH>/IMPROVE/workflows/deephyper_hpo/README.md
   ```

For any further questions or issues, please refer to the project documentation or contact the project maintainers.

The output should be of the style

```
....
.....
├── 0.9
│   ├── logs.txt
│   ├── model
│   │   ├── assets
│   │   ├── keras_metadata.pb
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── param_log_file.txt
│   ├── val_scores.json
│   └── val_y_data_predicted.csv
├── context.yaml
├── hpo_results.csv
├── model_0.0.pkl
├── model_0.10.pkl
├── model_0.11.pkl
├── model_0.12.pkl
├── model_0.13.pkl
├── model_0.1.pkl
├── model_0.2.pkl
├── model_0.3.pkl
├── model_0.4.pkl
├── model_0.5.pkl
├── model_0.6.pkl
├── model_0.7.pkl
├── model_0.8.pkl
├── model_0.9.pkl
└── results.csv

```