import time
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import textwrap

# [Req] IMPROVE imports
import improvelib.utils as frm
import improvelib.applications.drug_response_prediction.drp_utils as drp
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig

# Model-specifc imports
from model_params_def import preprocess_params # [Req]
from uno_utils_improve import print_duration, subset_data

filepath = Path(__file__).resolve().parent  # [Req]


def run(params: Dict):
    """ Run data preprocessing.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the preprocessed (generated)
            ML data files.
    """
    # Record start time
    preprocess_start_time = time.time()
    
    # ------------------------------------------------------
    # [Req] Validity check of feature representations
    # ------------------------------------------------------
    # not needed for this data/model

    # ------------------------------------------------------
    # [Req] Determine preprocessing on training data
    # ------------------------------------------------------

    # Reading hyperparameters
    preprocess_debug = params["preprocess_debug"]
    preprocess_subset_data = params["preprocess_subset_data"]

    temp_start_time = time.time()
    
    # [Req] Load omics data
    print("\nLoading omics data.")
    ge = frm.get_x_data(file = params['cell_transcriptomic_file'], 
                                        benchmark_dir = params['input_dir'], 
                                        column_name = params['canc_col_name'])

    # [Req] Load drug data
    print("\nLoading drugs data.")
    md = frm.get_x_data(file = params['drug_mordred_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])

    temp_end_time = time.time()
    print("")
    print_duration("Loading Data", temp_start_time, temp_end_time)

    if preprocess_debug:
        print("Loaded Gene Expression:")
        print(ge.head())
        print(ge.shape)
        print("")
        print("Loaded Mordred Descriptors:")
        print(md.head())
        print(md.shape)
        print("")

    temp_start_time = time.time()
    # Prepare data to fit feature scaler
    print("Load train response data.")
    response_train = frm.get_y_data(split_file=params["train_split_file"], 
                                   benchmark_dir=params['input_dir'], 
                                   y_data_file=params['y_data_file'])
    response_train = response_train.dropna(subset=[params['y_col_name']])
    response_shape_before_merge = response_train.shape
    print("Find intersection of training data.")
    response_train = frm.get_y_data_with_features(response_train, ge, params['canc_col_name'])
    response_train = frm.get_y_data_with_features(response_train, md, params['drug_col_name'])
    ge_train = frm.get_features_in_y_data(ge, response_train, params['canc_col_name'])
    md_train = frm.get_features_in_y_data(md, response_train, params['drug_col_name'])

    if preprocess_debug:
        print(textwrap.dedent(f"""
            Gene Expression Shape Before Subsetting With Response: {ge.shape}
            Gene Expression Shape After Subsetting With Response: {ge_train.shape}
            Mordred Shape Before Subsetting With Response: {md.shape}
            Mordred Shape After Subsetting With Response: {md_train.shape}
            Response Shape Before Merging With Data: {response_shape_before_merge}
            Response Shape After Merging With Data: {response_train.shape}
        """))

    # Create feature scaler
    print("Determine transformations.")
    frm.determine_transform(ge_train, 'ge_transform', params['cell_transcriptomic_transform'], params['output_dir'])
    frm.determine_transform(md_train, 'md_transform', params['drug_mordred_transform'], params['output_dir'])

    del response_train, ge_train, md_train
    temp_end_time = time.time()
    print_duration("Creating Scalers", temp_start_time, temp_end_time)

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    stages = {
        "train": params["train_split_file"],
        "val": params["val_split_file"],
        "test": params["test_split_file"],
    }

    for stage, split_file in stages.items():
        split_start_time = time.time()
        print(f"Prepare data for stage {stage}.")
        print(f"Find intersection of {stage} data.")
        response_stage = frm.get_y_data(split_file=split_file, 
                                benchmark_dir=params['input_dir'], 
                                y_data_file=params['y_data_file'])
        response_stage = response_stage.dropna(subset=[params['y_col_name']])
        response_shape_before_merge = response_stage.shape
        response_stage = frm.get_y_data_with_features(response_stage, ge, params['canc_col_name'])
        response_stage = frm.get_y_data_with_features(response_stage, md, params['drug_col_name'])
        ge_stage = frm.get_features_in_y_data(ge, response_stage, params['canc_col_name'])
        md_stage = frm.get_features_in_y_data(md, response_stage, params['drug_col_name'])
        
        if preprocess_debug:
            print(textwrap.dedent(f"""
                Gene Expression Shape Before Subsetting With Response: {ge.shape}
                Gene Expression Shape After Subsetting With Response: {ge_stage.shape}
                Mordred Shape Before Subsetting With Response: {md.shape}
                Mordred Shape After Subsetting With Response: {md_stage.shape}
                Response Shape Before Merging With Data: {response_shape_before_merge}
                Response Shape After Merging With Data: {response_stage.shape}
            """))

        temp_start_time = time.time()
        print(f"Transform {stage} data.")
        ge_stage = frm.transform_data(ge_stage, 'ge_transform', params['output_dir'])
        md_stage = frm.transform_data(md_stage, 'md_transform', params['output_dir'])
        temp_end_time = time.time()
        print_duration(f"Applying Scaler to {stage.capitalize()}", temp_start_time, temp_end_time)

        if preprocess_debug:
            print("Gene Expression Scaled:")
            print(ge_stage.head())
            print(ge_stage.shape)
            print("")
            print("Mordred Descriptors Scaled:")
            print(md_stage.head())
            print(md_stage.shape)
            print("")

        if preprocess_subset_data:
            total_num_samples = 5000
            stage_proportions = {"train": 0.8, "val": 0.1, "test": 0.1}
            response_stage = subset_data(response_stage, stage, total_num_samples, stage_proportions)

        temp_start_time = time.time()
        print(f"Saving {stage.capitalize()} Data (unmerged) to Parquet")
        
        # [Req] Build data name
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)
        ge_fname = f"ge_{data_fname}"
        md_fname = f"md_{data_fname}"
        rsp_fname = f"rsp_{data_fname}"
        
        ge_stage = ge_stage.reset_index()
        ge_stage["improve_sample_id"] = ge_stage['improve_sample_id'].astype(str)
        first_column = ge_stage.iloc[:, :1]
        rest_columns = ge_stage.iloc[:, 1:].add_prefix('ge.')
        ge_stage = pd.concat([first_column, rest_columns], axis=1)
        ge_stage.to_parquet(Path(params["output_dir"]) / ge_fname)
        
        md_stage = md_stage.reset_index()
        md_stage["improve_chem_id"] = md_stage['improve_chem_id'].astype(str)
        md_stage.to_parquet(Path(params["output_dir"]) / md_fname)
        
        response_stage.to_parquet(Path(params["output_dir"]) / rsp_fname)
        
        # [Req] Save y dataframe for the current stage
        frm.save_stage_ydf(response_stage, stage, params["output_dir"])
        
        temp_end_time = time.time()
        print_duration(f"Saving {stage.capitalize()} Dataframes", temp_start_time, temp_end_time)

        split_end_time = time.time()
        print_duration(f"Processing {stage.capitalize()} Data", split_start_time, split_end_time)

    preprocess_end_time = time.time()
    print_duration(
        f"Preprocessing Data (All)", preprocess_start_time, preprocess_end_time
    )

    return params["output_dir"]

# [Req]
def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="uno_params.ini",
        additional_definitions=preprocess_params,
        required=None,
    )
    timer_preprocess = frm.Timer()
    ml_data_outdir = run(params)
    timer_preprocess.save_timer(dir_to_save=params["output_dir"], 
                                filename='runtime_preprocess.json', 
                                extra_dict={"stage": "preprocess"})
    print("\nFinished data preprocessing.")

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
