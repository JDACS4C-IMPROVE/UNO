""" Preprocessing of raw data to generate datasets for UNO Model. """
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Union
from uno_utils_improve import print_duration, get_common_samples, get_column_ranges, subset_data

# Script Dependencies: pandas, numpy, joblib, scikit-learn

# [Req] Import params
from params import app_preproc_params, model_preproc_params, app_train_params, model_train_params

import numpy as np
import pandas as pd
import joblib
import textwrap

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    PowerTransformer,
)

filepath = Path(__file__).resolve().parent  # [Req]

# IMPROVE imports
import improvelib.utils as frm
import improvelib.applications.drug_response_prediction.drp_utils as drp
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils

# ---------------------
# [Req] Parameter lists
# ---------------------
preprocess_params = app_preproc_params + model_preproc_params
train_params = app_train_params + model_train_params

def scale_df(
    df: pd.DataFrame, scaler_name: str = "std", scaler=None, verbose: bool = False
):
    """Returns a dataframe with scaled data."""
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return df, None

    # Scale data
    df_num = df.select_dtypes(include="number")

    if scaler is None:  # Create scikit scaler object
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "maxabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        elif scaler_name in ["l1", "l2", "max"]:
            scaler = Normalizer(norm=scaler_name)
        elif scaler_name == "power_yj":
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            print(
                f"The specified scaler ({scaler_name}) is not implemented (no df scaling)."
            )
            return df, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else:  # Apply passed scikit scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    df[df_num.columns] = df_norm

    # Remove rows with NaN or inf values and print proportion of rows removed
    rows_before = df.shape[0]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    rows_after = df.shape[0]
    proportion_removed = (rows_before - rows_after) / rows_before
    print(f"Proportion of rows removed for corrupted data: {proportion_removed:.3%}")

    return df, scaler

def run(params: Dict):
    """Execute data pre-processing for UNO model."""
    # Record start time
    preprocess_start_time = time.time()

    # [Req] Build paths and create ML data dir
    params = frm.build_paths(params)
    frm.create_outdir(outdir=params["output_dir"])

    # Reading hyperparameters
    preprocess_debug = params["preprocess_debug"]
    preprocess_subset_data = params["preprocess_subset_data"]

    temp_start_time = time.time()
    # [Req] Load omics data
    print("\nLoading omics data.")
    omics_obj = omics_utils.OmicsLoader(params)
    ge = omics_obj.dfs['cancer_gene_expression.tsv']
    ge["improve_sample_id"] = ge['improve_sample_id'].astype(str)
    first_column = ge.iloc[:, :1]
    rest_columns = ge.iloc[:, 1:].add_prefix('ge.')
    ge = pd.concat([first_column, rest_columns], axis=1)

    # [Req] Load drug data
    print("\nLoading drugs data.")
    drugs_obj = drugs_utils.DrugsLoader(params)
    md = drugs_obj.dfs['drug_mordred.tsv']
    md = md.reset_index()
    md["improve_chem_id"] = md['improve_chem_id'].astype(str)

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
    # Data prep to create scaler on
    rsp_tr = drp.DrugResponseLoader(
        params, split_file=params["train_split_file"], verbose=False
    ).dfs["response.tsv"]
    rsp_vl = drp.DrugResponseLoader(
        params, split_file=params["val_split_file"], verbose=False
    ).dfs["response.tsv"]
    rsp = pd.concat([rsp_tr, rsp_vl], axis=0)
    rsp = rsp[[params["canc_col_name"], params["drug_col_name"], params["y_col_name"]]]

    ge_sub, md_sub, rsp_sub = get_common_samples(
        ge, md, rsp, params["canc_col_name"], params["drug_col_name"]
    )

    if preprocess_debug:
        print(textwrap.dedent(f"""
            Gene Expression Shape Before Subsetting With Response: {ge.shape}
            Gene Expression Shape After Subsetting With Response: {ge_sub.shape}
            Mordred Shape Before Subsetting With Response: {md.shape}
            Mordred Shape After Subsetting With Response: {md_sub.shape}
            Response Shape Before Merging With Data: {rsp.shape}
            Response Shape After Merging With Data: {rsp_sub.shape}
        """))

    # Create Feature Scaler
    print("\nCreating Feature Scalers\n")
    _, ge_scaler = scale_df(ge_sub, scaler_name=params["ge_scaling"])
    ge_scaler_fpath = Path(params["output_dir"]) / params["ge_scaler_fname"]
    joblib.dump(ge_scaler, ge_scaler_fpath)
    print("Scaler object for gene expression: ", ge_scaler_fpath)

    _, md_scaler = scale_df(md_sub, scaler_name=params["md_scaling"])
    md_scaler_fpath = Path(params["output_dir"]) / params["md_scaler_fname"]
    joblib.dump(md_scaler, md_scaler_fpath)
    print("Scaler object for Mordred:         ", md_scaler_fpath)

    del rsp, rsp_tr, rsp_vl, ge_sub, md_sub
    temp_end_time = time.time()
    print_duration("Creating Scalers", temp_start_time, temp_end_time)

    # [Req] Construct ML data for every stage (train, val, test)
    stages = {
        "train": params["train_split_file"],
        "val": params["val_split_file"],
        "test": params["test_split_file"],
    }

    for stage, split_file in stages.items():
        split_start_time = time.time()
        print(f"Stage: {stage.upper()}")
        rsp = drp.DrugResponseLoader(params, split_file=split_file, verbose=False).dfs[
            "response.tsv"
        ]
        rsp = rsp[[params["canc_col_name"], params["drug_col_name"], params["y_col_name"]]]

        ge_sub, md_sub, rsp_sub = get_common_samples(
            ge, md, rsp, params["canc_col_name"], params["drug_col_name"]
        )

        if preprocess_debug:
            print(textwrap.dedent(f"""
                Gene Expression Shape Before Subsetting With Response: {ge.shape}
                Gene Expression Shape After Subsetting With Response: {ge_sub.shape}
                Mordred Shape Before Subsetting With Response: {md.shape}
                Mordred Shape After Subsetting With Response: {md_sub.shape}
                Response Shape Before Merging With Data: {rsp.shape}
                Response Shape After Merging With Data: {rsp_sub.shape}
            """))

        temp_start_time = time.time()
        print("\nScaling data")
        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler)
        md_sc, _ = scale_df(md_sub, scaler=md_scaler)
        temp_end_time = time.time()
        print_duration(f"Applying Scaler to {stage.capitalize()}", temp_start_time, temp_end_time)

        if preprocess_debug:
            print("Gene Expression Scaled:")
            print(ge_sc.head())
            print(ge_sc.shape)
            print("")
            print("Mordred Descriptors Scaled:")
            print(md_sc.head())
            print(md_sc.shape)
            print("")

        if preprocess_subset_data:
            total_num_samples = 5000
            stage_proportions = {"train": 0.8, "val": 0.1, "test": 0.1}
            rsp_sub = subset_data(rsp_sub, stage, total_num_samples, stage_proportions)

        temp_start_time = time.time()
        print(f"Saving {stage.capitalize()} Data (unmerged) to Parquet")
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)
        ge_fname = f"ge_{data_fname}"
        md_fname = f"md_{data_fname}"
        rsp_fname = f"rsp_{data_fname}"
        ge_sc.to_parquet(Path(params["output_dir"]) / ge_fname)
        md_sc.to_parquet(Path(params["output_dir"]) / md_fname)
        rsp_sub.to_parquet(Path(params["output_dir"]) / rsp_fname)
        ydf = rsp_sub[['improve_sample_id', 'improve_chem_id', params["y_col_name"]]]
        frm.save_stage_ydf(ydf, stage, params["output_dir"])
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
    # [Req]
    additional_definitions = preprocess_params
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="uno_default_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    ml_data_outdir = run(params)
    print(
        "\nFinished UNO pre-processing (transformed raw DRP data to model input ML data)."
    )

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])