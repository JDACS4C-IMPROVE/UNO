import time
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
import improvelib.utils as frm  # Utility functions

# Model-specifc imports
from model_params_def import infer_params # [Req]
from uno_utils_improve import (
    data_merge_generator, batch_predict
)

# [Req]
filepath = Path(__file__).resolve().parent  

# ------------------------------------------------------
# [Req] Check GPU availability
# ------------------------------------------------------
gpus = tf.config.list_logical_devices('GPU')

if gpus:
    print(f"TensorFlow will use the GPU by default: {[gpu.name for gpu in gpus]}")
else:
    print("No GPU available. TensorFlow will use the CPU.")


# ------------------------------------------------------
# [Req] Run inference with trained model
# ------------------------------------------------------
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
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")
    test_ge_fname = f"ge_{test_data_fname}"
    test_md_fname = f"md_{test_data_fname}"
    test_rsp_fname = f"rsp_{test_data_fname}"

    # Load test data from input directory
    ts_ge = pd.read_parquet(Path(params["input_data_dir"]) / test_ge_fname)
    ts_md = pd.read_parquet(Path(params["input_data_dir"]) / test_md_fname)
    ts_rsp = pd.read_parquet(Path(params["input_data_dir"]) / test_rsp_fname)
    ts_rsp = ts_rsp[[params["canc_col_name"], params["drug_col_name"], params["y_col_name"]]]

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
    print("Loading model: '%s'" % modelpath)
    try:
        model = load_model(modelpath)
    except IOError as e:
        print("Loading model failed: " + str(e))
        exit(1)

    # Create data generator for batch predictions
    generator_batch_size = params["infer_batch"]
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
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=test_true, 
            y_pred=test_pred, 
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


# [Req]
def main(args):
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="uno_params.ini",
                                       additional_definitions=infer_params)
    timer_infer = frm.Timer()    
    status = run(params)
    timer_infer.save_timer(dir_to_save=params["output_dir"], 
                           filename='runtime_infer.json', 
                           extra_dict={"stage": "infer"})
    print("\nFinished model inference.")


if __name__ == "__main__":
    # Record the start time for inference
    infer_start_time = time.time()
    main(sys.argv[1:])