import time
import os
import sys
from pathlib import Path
from typing import Dict

# Import required modules from improvelib
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm  # Utility functions

# Additional third-party library imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import custom modules from local scripts
from uno_preprocess_improve import preprocess_params
from uno_train_improve import metrics_list, train_params
from uno_utils_improve import (
    data_merge_generator, batch_predict, print_duration, clean_arrays,
    check_array, calculate_sstot
)

# Set filepath to the directory where the script is located
filepath = Path(__file__).resolve().parent  

# ---------------------
# Parameter Lists
# ---------------------
# Define two parameter lists required by the inference process:
# 1. App-specific parameters for monotherapy drug response prediction.
# 2. Model-specific parameters (optional; LightGBM in this case).

# Currently no app-specific parameters.
app_infer_params = []

# Optional model-specific parameters.
model_infer_params = []

# Combine both parameter lists to pass to frm.initialize_parameters() in the main().
infer_params = app_infer_params + model_infer_params


def run(params: Dict):
    """
    Run model inference and compute prediction scores.

    Args:
        params (dict): Dictionary containing model and application parameters.

    Returns:
        bool: True if inference completes successfully.
    """
    # ------------------------------------------------------
    # Create filenames and load test set data
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="tcga_test")
    test_ge_fname = f"ge_{test_data_fname}"
    test_md_fname = f"md_{test_data_fname}"
    test_rsp_fname = f"rsp_{test_data_fname}"

    # Load test data from input directory
    ts_ge = pd.read_parquet(Path(params["input_data_dir"]) / test_ge_fname)
    ts_md = pd.read_parquet(Path(params["input_data_dir"]) / test_md_fname)
    ts_rsp = pd.read_parquet(Path(params["input_data_dir"]) / test_rsp_fname)

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Build the model path
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"]
    )
    # Load the pre-trained model
    print("loading model: '%s'" % modelpath)
    try:
        model = load_model(modelpath)
    except IOError as e:
        print("model load failed: " + str(e))
        exit(1)

    # Create data generator for batch predictions
    generator_batch_size = params["generator_batch_size"]
    test_steps = int(np.ceil(len(ts_rsp) / generator_batch_size))
    test_gen = data_merge_generator(
        ts_rsp, ts_ge, ts_md, generator_batch_size, 
        params, merge_preserve_order=True, verbose=False
    )

    # Perform batch predictions
    test_pred, test_true = batch_predict(model, test_gen, test_steps)

    # ------------------------------------------------------
    # Save raw predictions to a dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=test_true, 
        y_pred=test_pred, 
        stage="tcga_test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # ------------------------------------------------------
    # Compute and save performance scores (optional)
    # ------------------------------------------------------
    if params.get("calc_infer_scores", False):
        test_scores = frm.compute_performance_scores(
            y_true=test_true, 
            y_pred=test_pred, 
            stage="tcga_test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


def main(args):
    """
    Main function to initialize parameters and run inference.

    Args:
        args (list): Command-line arguments.
    """
    # Combine parameter definitions from preprocessing, training, and inference stages
    additional_definitions = preprocess_params + train_params + infer_params

    # Initialize inference configuration
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="tcga_params_uno.txt",
        additional_definitions=additional_definitions,
        required=None
    )

    # Run model inference
    test_scores = run(params)

    # Record inference duration
    infer_end_time = time.time()
    print_duration("Infering", infer_start_time, infer_end_time)
    print("\nFinished model inference.")


if __name__ == "__main__":
    # Record the start time for inference
    infer_start_time = time.time()
    main(sys.argv[1:])