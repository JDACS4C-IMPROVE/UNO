"""
Model-specific params
"""
from improvelib.utils import str2bool

preprocess_params = [
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


train_params = [
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
        "name": "preprocess_subset_data",
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

infer_params = []
