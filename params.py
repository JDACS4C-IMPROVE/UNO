# [Req] IMPROVE/CANDLE imports
# from improve import framework as frm
# import improvelib.utils as frm
from improvelib.utils import str2bool

# --------------------------------
# [Req] Preprocess Parameter Lists
# --------------------------------
# Two parameter lists are required:
# 1. app_preproc_params
# 2. model_preproc_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
app_preproc_params = [
    {
        "name": "y_data_files",  # default
        "type": str,
        "help": "List of files that contain the y (prediction variable) data. \
             Example: [['response.tsv']]",
    },
    {
        "name": "x_data_canc_files",  # required
        "type": str,
        "help": "List of feature files including gene_system_identifer. Examples: \n\
             1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
             2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
    },
    {
        "name": "x_data_drug_files",  # required
        "type": str,
        "help": "List of feature files. Examples: \n\
             1) [['drug_SMILES.tsv']] \n\
             2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
    },
    {
        "name": "canc_col_name",
        "default": "improve_sample_id",  # default
        "type": str,
        "help": "Column name in the y (response) data file that contains the cancer sample ids.",
    },
    {
        "name": "drug_col_name",  # default
        "default": "improve_chem_id",
        "type": str,
        "help": "Column name in the y (response) data file that contains the drug ids.",
    },
]

# 2. Model-specific params (Model: Uno)
model_preproc_params = [
    {
        "name": "use_lincs",
        "type": str2bool,
        "default": False,
        "help": "Flag to indicate if using landmark genes.",
    },
    {
        "name": "ge_scaling",
        "type": str,
        "default": "std",
        "choice": ["std", "minmax", "maxabs", "robust", "l1", "l2", "max", "power_yj"],
        "help": "Scaler for gene expression data.",
    },
    {
        "name": "ge_scaler_fname",
        "type": str,
        "default": "x_data_gene_expression_scaler.gz",
        "help": "File name to save the gene expression scaler object.",
    },
    {
        "name": "md_scaling",
        "type": str,
        "default": "std",
        "choice": ["std", "minmax", "miabs", "robust", "l1", "l2", "max", "power_yj"],
        "help": "Scaler for gene expression data.",
    },
    {
        "name": "md_scaler_fname",
        "type": str,
        "default": "x_data_mordred_scaler.gz",
        "help": "File name to save the Mordred scaler object.",
    },
    {
        "name": "preprocess_debug",
        "type": bool,
        "default": False,
        "help": "Debug mode to show data",
    },
    {
        "name": "preprocess_subset_data",
        "type": bool,
        "default": False,
        "help": "Subsetting data for faster test runs",
    },
]



# ---------------------------
# [Req] Train Parameter Lists
# ---------------------------
# Two parameter lists are required:
# 1. app_train_params
# 2. model_train_params
#
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params for this script.
app_train_params = []

# 2. Model-specific params (Model: UNO)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.
model_train_params = [
    {
        "name": "epochs",
        "type": int,
        "default": 150,
        "help": "Number of epochs in training.",
    },
    {
        "name": "batch_size",
        "type": int,
        "default": 32,
        "help": "Batch size for training.",
    },
    {
        "name": "generator_batch_size",
        "type": int,
        "default": 1024,
        "help": "Batch size for prediction.",
    },
    {
        "name": "learning_rate",
        "type": float,
        "default": 1e-5,
        "help": "learning rate.",
    },
    {
        "name": "warmup_epochs",
        "type": int,
        "default": 5,
        "help": "Number of warmup epochs.",
    },
    {
        "name": "warmup_type",
        "type": str,
        "default": "quadratic",
        "help": "Type of warmup for learning rate.",
    },
    {
        "name": "reduce_lr_patience",
        "type": int,
        "default": 3,
        "help": "Patience epochs for reducing learning rate.",
    },
    {
        "name": "reduce_lr_factor",
        "type": float,
        "default": 0.5,
        "help": "Factor for reducing learning rate after plateau.",
    },
    {
        "name": "optimizer",
        "type": str,
        "default": "Adam",
        "help": "Optimizer for gradient descent.",
    },
    {
        "name": "regression_activation",
        "type": str,
        "default": "sigmoid",
        "help": "Output activation function since target is [0,1]",
    },
    {
        "name": "loss",
        "type": str,
        "default": "mse",
        "help": "Loss function to be used.",
    },
    {
        "name": "early_stop_metric",
        "type": str,
        "default": "mse",
        "help": "Loss function for early stopping",
    },
    {
        "name": "early_stopping_patience",
        "type": int,
        "default": 20,
        "help": "Patience for early stopping training after no improvement",
    },
    {
        "name": "train_debug",
        "type": bool,
        "default": False,
        "help": "Debug mode to show training information",
    },
    {
        "name": "train_subset_data",
        "type": bool,
        "default": False,
        "help": "Subsetting data for faster test runs",
    },
    {
        "name": "hyperparam_space",
        "type": str,
        "default": "global",
        "help": "Defines the hyperparameter space to use. Could be global, by block, or by layer"
    },


    {
        "name": "dropout",
        "type": float,
        "default": 0.1,
        "help": "Global dropout rate."
    },
    {
        "name": "activation",
        "type": str,
        "default": "relu",
        "help": "Global activation function."
    },


    {
        "name": "canc_arch",
        "type": int,
        "default": [1000, 1000, 1000],
        "help": "Block architecture for cancer layers."
    },
    {
        "name": "canc_activation",
        "type": str,
        "default": "relu",
        "help": "Block activation function for cancer layers."
    },
    {
        "name": "canc_dropout",
        "type": float,
        "default": 0.1,
        "help": "Block dropout rate for cancer layers."
    },
    {
        "name": "drug_arch",
        "type": int,
        "default": [1000, 1000, 1000],
        "help": "Block architecture for drug layers."
    },
    {
        "name": "drug_activation",
        "type": str,
        "default": "relu",
        "help": "Block activation function for drug layers."
    },
    {
        "name": "drug_dropout",
        "type": float,
        "default": 0.1,
        "help": "Block dropout rate for drug layers."
    },
    {
        "name": "interaction_arch",
        "type": int,
        "default": [1000, 1000, 1000, 1000, 1000],
        "help": "Block architecture for interaction layers."
    },
    {
        "name": "interaction_activation",
        "type": str,
        "default": "relu",
        "help": "Block activation function for interaction layers."
    },
    {
        "name": "interaction_dropout",
        "type": float,
        "default": 0.1,
        "help": "Block dropout rate for interaction layers."
    },


    {
        "name": "canc_num_layers",
        "type": int,
        "default": 3,
        "help": """
                Number of cancer feature layers. The
                script reads layer sizes, dropouts, and
                activation up to number of layers specified.
                """,
    },
    {
        "name": "canc_layer_1_size",
        "type": int,
        "default": 1000,
        "help": "Size of first cancer feature layer.",
    },
    {
        "name": "canc_layer_2_size",
        "type": int,
        "default": 1000,
        "help": "Size of second cancer feature layer.",
    },
    {
        "name": "canc_layer_3_size",
        "type": int,
        "default": 1000,
        "help": "Size of third cancer feature layer.",
    },
    {
        "name": "canc_layer_4_size",
        "type": int,
        "default": 512,
        "help": "Size of fourth cancer feature layer.",
    },
    {
        "name": "canc_layer_5_size",
        "type": int,
        "default": 256,
        "help": "Size of fifth cancer feature layer.",
    },
    {
        "name": "canc_layer_6_size",
        "type": int,
        "default": 128,
        "help": "Size of sixth cancer feature layer.",
    },
    {
        "name": "canc_layer_7_size",
        "type": int,
        "default": 64,
        "help": "Size of seventh cancer feature layer.",
    },
    {
        "name": "canc_layer_8_size",
        "type": int,
        "default": 32,
        "help": "Size of eighth cancer feature layer.",
    },
    {
        "name": "canc_layer_9_size",
        "type": int,
        "default": 32,
        "help": "Size of ninth cancer feature layer.",
    },
    {
        "name": "canc_layer_1_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for first cancer feature layer.",
    },
    {
        "name": "canc_layer_2_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for second cancer feature layer.",
    },
    {
        "name": "canc_layer_3_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for third cancer feature layer.",
    },
    {
        "name": "canc_layer_4_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fourth cancer feature layer.",
    },
    {
        "name": "canc_layer_5_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fifth cancer feature layer.",
    },
    {
        "name": "canc_layer_6_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for sixth cancer feature layer.",
    },
    {
        "name": "canc_layer_7_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for seventh cancer feature layer.",
    },
    {
        "name": "canc_layer_8_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for eighth cancer feature layer.",
    },
    {
        "name": "canc_layer_9_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for ninth cancer feature layer.",
    },
    {
        "name": "canc_layer_1_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for first cancer feature layer.",
    },
    {
        "name": "canc_layer_2_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for second cancer feature layer.",
    },
    {
        "name": "canc_layer_3_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for third cancer feature layer.",
    },
    {
        "name": "canc_layer_4_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for fourth cancer feature layer.",
    },
    {
        "name": "canc_layer_5_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for fifth cancer feature layer.",
    },
    {
        "name": "canc_layer_6_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for sixth cancer feature layer.",
    },
    {
        "name": "canc_layer_7_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for seventh cancer feature layer.",
    },
    {
        "name": "canc_layer_8_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for eighth cancer feature layer.",
    },
    {
        "name": "canc_layer_9_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for ninth cancer feature layer.",
    },
    {
        "name": "drug_num_layers",
        "type": int,
        "default": 3,
        "help": """
                Number of drug feature layers. The script
                reads layer sizes, dropouts, and activation
                up to number of layers specified.
                """,
    },
    {
        "name": "drug_layer_1_size",
        "type": int,
        "default": 1000,
        "help": "Size of first drug feature layer.",
    },
    {
        "name": "drug_layer_2_size",
        "type": int,
        "default": 1000,
        "help": "Size of second drug feature layer.",
    },
    {
        "name": "drug_layer_3_size",
        "type": int,
        "default": 1000,
        "help": "Size of third drug feature layer.",
    },
    {
        "name": "drug_layer_4_size",
        "type": int,
        "default": 512,
        "help": "Size of fourth drug feature layer.",
    },
    {
        "name": "drug_layer_5_size",
        "type": int,
        "default": 256,
        "help": "Size of fifth drug feature layer.",
    },
    {
        "name": "drug_layer_6_size",
        "type": int,
        "default": 128,
        "help": "Size of sixth drug feature layer.",
    },
    {
        "name": "drug_layer_7_size",
        "type": int,
        "default": 64,
        "help": "Size of seventh drug feature layer.",
    },
    {
        "name": "drug_layer_8_size",
        "type": int,
        "default": 32,
        "help": "Size of eighth drug feature layer.",
    },
    {
        "name": "drug_layer_9_size",
        "type": int,
        "default": 32,
        "help": "Size of ninth drug feature layer.",
    },
    {
        "name": "drug_layer_1_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for first drug feature layer.",
    },
    {
        "name": "drug_layer_2_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for second drug feature layer.",
    },
    {
        "name": "drug_layer_3_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for third drug feature layer.",
    },
    {
        "name": "drug_layer_4_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fourth drug feature layer.",
    },
    {
        "name": "drug_layer_5_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fifth drug feature layer.",
    },
    {
        "name": "drug_layer_6_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for sixth drug feature layer.",
    },
    {
        "name": "drug_layer_7_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for seventh drug feature layer.",
    },
    {
        "name": "drug_layer_8_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for eighth drug feature layer.",
    },
    {
        "name": "drug_layer_9_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for ninth drug feature layer.",
    },
    {
        "name": "drug_layer_1_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for first drug feature layer.",
    },
    {
        "name": "drug_layer_2_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for second drug feature layer.",
    },
    {
        "name": "drug_layer_3_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for third drug feature layer.",
    },
    {
        "name": "drug_layer_4_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for fourth drug feature layer.",
    },
    {
        "name": "drug_layer_5_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for fifth drug feature layer.",
    },
    {
        "name": "drug_layer_6_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for sixth drug feature layer.",
    },
    {
        "name": "drug_layer_7_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for seventh drug feature layer.",
    },
    {
        "name": "drug_layer_8_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for eighth drug feature layer.",
    },
    {
        "name": "drug_layer_9_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for ninth drug feature layer.",
    },
    {
        "name": "interaction_num_layers",
        "type": int,
        "default": 5,
        "help": """
                Number of interaction feature layers. The
                script reads layer sizes, dropouts, and
                activation up to number of layers specified.
                """,
    },
    {
        "name": "interaction_layer_1_size",
        "type": int,
        "default": 1000,
        "help": "Size of first interaction layer.",
    },
    {
        "name": "interaction_layer_2_size",
        "type": int,
        "default": 1000,
        "help": "Size of second interaction layer.",
    },
    {
        "name": "interaction_layer_3_size",
        "type": int,
        "default": 1000,
        "help": "Size of third interaction layer.",
    },
    {
        "name": "interaction_layer_4_size",
        "type": int,
        "default": 1000,
        "help": "Size of fourth interaction layer.",
    },
    {
        "name": "interaction_layer_5_size",
        "type": int,
        "default": 1000,
        "help": "Size of fifth interaction layer.",
    },
    {
        "name": "interaction_layer_6_size",
        "type": int,
        "default": 512,
        "help": "Size of sixth interaction layer.",
    },
    {
        "name": "interaction_layer_7_size",
        "type": int,
        "default": 256,
        "help": "Size of seventh interaction layer.",
    },
    {
        "name": "interaction_layer_8_size",
        "type": int,
        "default": 128,
        "help": "Size of eighth interaction layer.",
    },
    {
        "name": "interaction_layer_9_size",
        "type": int,
        "default": 64,
        "help": "Size of ninth interaction layer.",
    },
    {
        "name": "interaction_layer_1_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for first interaction layer.",
    },
    {
        "name": "interaction_layer_2_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for second interaction layer.",
    },
    {
        "name": "interaction_layer_3_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for third interaction layer.",
    },
    {
        "name": "interaction_layer_4_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fourth interaction layer.",
    },
    {
        "name": "interaction_layer_5_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fifth interaction layer.",
    },
    {
        "name": "interaction_layer_6_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for sixth interaction layer.",
    },
    {
        "name": "interaction_layer_7_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for seventh interaction layer.",
    },
    {
        "name": "interaction_layer_8_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for eighth interaction layer.",
    },
    {
        "name": "interaction_layer_9_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for ninth interaction layer.",
    },
    {
        "name": "interaction_layer_1_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for first interaction layer.",
    },
    {
        "name": "interaction_layer_2_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for second interaction layer.",
    },
    {
        "name": "interaction_layer_3_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for third interaction layer.",
    },
    {
        "name": "interaction_layer_4_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for fourth interaction layer.",
    },
    {
        "name": "interaction_layer_5_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for fifth interaction layer.",
    },
    {
        "name": "interaction_layer_6_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for sixth interaction layer.",
    },
    {
        "name": "interaction_layer_7_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for seventh interaction layer.",
    },
    {
        "name": "interaction_layer_8_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for eighth interaction layer.",
    },
    {
        "name": "interaction_layer_9_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for ninth interaction layer.",
    }

]
