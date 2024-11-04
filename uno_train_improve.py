import time
import os
import sys
from pathlib import Path
from typing import Dict

# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dropout, Lambda
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    LearningRateScheduler,
    EarlyStopping,
)

# Custom imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# Import parameters
from params import app_preproc_params, model_preproc_params, app_train_params, model_train_params

# Import custom utility functions
from uno_utils_improve import (
    data_merge_generator, 
    batch_predict, 
    print_duration, 
    get_optimizer,
    subset_data,
    calculate_sstot, 
    R2Callback_efficient, 
    R2Callback_accurate, 
    warmup_scheduler,
    clean_arrays, check_array
)

# Check TensorFlow and GPU
print("TensorFlow Version:")
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# For TensorFlow 2.16, this would be the integer 16:
tf_minor_version = int(tf.__version__.split(".")[1])

# Get the current file path
filepath = Path(__file__).resolve().parent

# Define parameters
preprocess_params = app_preproc_params + model_preproc_params
train_params = app_train_params + model_train_params
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]

def read_architecture(params, hyperparam_space, arch_type):
    """Setup architecture for cancer, drug, and interaction layers."""
    layers_size = []
    layers_dropout = []
    layers_activation = []
    num_layers = 0
    if hyperparam_space == "global":
        if arch_type in ["canc", "drug"]:
            num_layers = 3
        elif arch_type == "interaction":
            num_layers = 5
        layers_size = [1000] * num_layers
        layers_dropout = [params["dropout"]] * num_layers
        layers_activation = [params["activation"]] * num_layers
    elif hyperparam_space == "block":
        num_layers = params[f"{arch_type}_num_layers"]
        arch = params[f"{arch_type}_arch"]
        layers_size = arch
        layers_dropout = [params[f"{arch_type}_dropout"]] * num_layers
        layers_activation = [params[f"{arch_type}_activation"]] * num_layers
    elif hyperparam_space == "layer":
        num_layers = params[f"{arch_type}_num_layers"]
        for i in range(num_layers):
            layers_size.append(params[f"{arch_type}_layer_{i+1}_size"])
            layers_dropout.append(params[f"{arch_type}_layer_{i+1}_dropout"])
            layers_activation.append(params[f"{arch_type}_layer_{i+1}_activation"])
    return num_layers, layers_size, layers_dropout, layers_activation

def run(params: Dict):
    """Run model training."""
    # Record start time
    train_start_time = time.time()

    # Create output directory and build model path
    frm.create_outdir(outdir=params["output_dir"])
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"]
    )

    # Read hyperparameters
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    generator_batch_size = params["generator_batch_size"]
    learning_rate = params["learning_rate"]
    max_lr = learning_rate * batch_size
    min_lr = max_lr / 10000
    warmup_epochs = params["warmup_epochs"]
    warmup_type = params["warmup_type"]
    initial_lr = max_lr / 100
    reduce_lr_factor = params["reduce_lr_factor"]
    reduce_lr_patience = params["reduce_lr_patience"]
    early_stopping_patience = params["early_stopping_patience"]
    optimizer = get_optimizer(params["optimizer"], initial_lr)
    train_debug = params["train_debug"]
    train_subset_data = params["train_subset_data"]
    preprocess_subset_data = params["preprocess_subset_data"]

    # Architecture Hyperparams
    hyperparam_space = params["hyperparam_space"]
    print(f"Hyperparam Space: {hyperparam_space}")

    # Read architecture for cancer, drug, and interaction layers
    canc_num_layers, canc_layers_size, canc_layers_dropout, canc_layers_activation = read_architecture(params, hyperparam_space, "canc")
    drug_num_layers, drug_layers_size, drug_layers_dropout, drug_layers_activation = read_architecture(params, hyperparam_space, "drug")
    interaction_num_layers, interaction_layers_size, interaction_layers_dropout, interaction_layers_activation = read_architecture(params, hyperparam_space, "interaction")

    # Final regression layer
    regression_activation = params["regression_activation"]

    if train_debug:
        print("CANCER LAYERS:", canc_layers_size, canc_layers_dropout, canc_layers_activation)
        print("DRUG LAYERS:", drug_layers_size, drug_layers_dropout, drug_layers_activation)
        print("INTERACTION LAYERS:", interaction_layers_size, interaction_layers_dropout, interaction_layers_activation)
        print("REGRESSION LAYER:", regression_activation)

    # Create file names and load data
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")
    train_ge_fname = f"ge_{train_data_fname}"
    train_md_fname = f"md_{train_data_fname}"
    train_rsp_fname = f"rsp_{train_data_fname}"
    tr_ge = pd.read_parquet(Path(params["input_dir"])/train_ge_fname)
    tr_md = pd.read_parquet(Path(params["input_dir"])/train_md_fname)
    tr_rsp = pd.read_parquet(Path(params["input_dir"])/train_rsp_fname)

    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")
    val_ge_fname = f"ge_{val_data_fname}"
    val_md_fname = f"md_{val_data_fname}"
    val_rsp_fname = f"rsp_{val_data_fname}"
    vl_ge = pd.read_parquet(Path(params["input_dir"])/val_ge_fname)
    vl_md = pd.read_parquet(Path(params["input_dir"])/val_md_fname)
    vl_rsp = pd.read_parquet(Path(params["input_dir"])/val_rsp_fname)

    if train_subset_data:
        total_num_samples = 5000
        stage_proportions = {"train": 0.8, "val": 0.1, "test": 0.1}
        tr_rsp = subset_data(tr_rsp, "Train", total_num_samples, stage_proportions)
        vl_rsp = subset_data(vl_rsp, "Validation", total_num_samples, stage_proportions)

    if train_debug:
        print("TRAIN DATA:", tr_rsp.head(), tr_rsp.shape)
        print("VAL DATA:", vl_rsp.head(), vl_rsp.shape)

    # Merge one row to get feature sets
    row = tr_rsp.iloc[0:1]
    merged_row = pd.merge(row, tr_ge, on=params["canc_col_name"], how="inner")
    merged_row = pd.merge(merged_row, tr_md, on=params["drug_col_name"], how="inner")
    if train_debug:
        print(merged_row.head(), merged_row.shape)

    num_ge_columns = len([col for col in merged_row.columns if col.startswith('ge')])
    num_md_columns = len([col for col in merged_row.columns if col.startswith('mordred')])

    # Define model inputs
    all_input = Input(shape=(num_ge_columns + num_md_columns,), name="all_input")
    canc_input = Lambda(lambda x: x[:, :num_ge_columns])(all_input)
    drug_input = Lambda(lambda x: x[:, num_ge_columns:num_ge_columns + num_md_columns])(all_input)

    # Define cancer expression input and encoding layers
    canc_encoded = canc_input
    for i in range(canc_num_layers):
        canc_encoded = Dense(canc_layers_size[i], activation=canc_layers_activation[i])(canc_encoded)
        canc_encoded = Dropout(canc_layers_dropout[i])(canc_encoded)

    # Define drug expression input and encoding layers
    drug_encoded = drug_input
    for i in range(drug_num_layers):
        drug_encoded = Dense(drug_layers_size[i], activation=drug_layers_activation[i])(drug_encoded)
        drug_encoded = Dropout(drug_layers_dropout[i])(drug_encoded)
    
    # Define interaction layers
    interaction_input = Concatenate()([canc_encoded, drug_encoded])
    interaction_encoded = interaction_input
    for i in range(interaction_num_layers):
        interaction_encoded = Dense(interaction_layers_size[i], activation=interaction_layers_activation[i])(interaction_encoded)
        interaction_encoded = Dropout(interaction_layers_dropout[i])(interaction_encoded)

    # Define final output layer
    output = Dense(1, activation=regression_activation)(interaction_encoded)

    # Compile model
    model = Model(inputs=all_input, outputs=output)
    model.compile(optimizer=optimizer, loss="mean_squared_error")

    if train_debug:
        model.summary()

    steps_per_epoch = int(np.ceil(len(tr_rsp) / batch_size))
    validation_steps = int(np.ceil(len(vl_rsp) / generator_batch_size))

    if tf_minor_version >= 16:
        learning_rate = model.optimizer.learning_rate
    else:
        learning_rate = model.optimizer.lr

    # Instantiate callbacks
    lr_scheduler = LearningRateScheduler(
        lambda epoch: warmup_scheduler(epoch, learning_rate, warmup_epochs, initial_lr, max_lr, warmup_type)
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    )

    train_ss_tot = calculate_sstot(tr_rsp[params["y_col_name"]])
    val_ss_tot = calculate_sstot(vl_rsp[params["y_col_name"]])

    r2_callback = R2Callback_efficient(
        train_ss_tot=train_ss_tot,
        val_ss_tot=val_ss_tot
    )

    epoch_start_time = time.time()

    # Create generators for training and validation
    train_gen = data_merge_generator(tr_rsp, tr_ge, tr_md, batch_size, params, shuffle=True, peek=True)
    val_gen = data_merge_generator(vl_rsp, vl_ge, vl_md, generator_batch_size, params, shuffle=False, peek=True)

    # Fit model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[r2_callback, lr_scheduler, reduce_lr, early_stopping],
    )

    epoch_end_time = time.time()
    total_epochs = len(history.history['loss'])
    global time_per_epoch 
    time_per_epoch = (epoch_end_time - epoch_start_time) / total_epochs

    # Save model
    model.save(modelpath)

    # Make predictions
    val_pred, val_true = batch_predict(
        model, 
        data_merge_generator(vl_rsp, vl_ge, vl_md, generator_batch_size, params, merge_preserve_order=True, verbose=False), 
        validation_steps
    )

    if (train_subset_data and preprocess_subset_data) or (not train_subset_data and not preprocess_subset_data):    
        frm.store_predictions_df(
            y_true=val_true, 
            y_pred=val_pred, 
            stage="val",
            y_col_name=params["y_col_name"],
            output_dir=params["output_dir"],
            input_dir=params["input_dir"]
        )

        val_scores = frm.compute_performance_scores(
            y_true=val_true, 
            y_pred=val_pred, 
            stage="val",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )
        
    return val_scores

def main(args):
    train_start_time = time.time()
    """Main function to run the model training."""
    additional_definitions = preprocess_params + train_params
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="uno_default_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    val_scores = run(params)
    train_end_time = time.time()
    print_duration("One epoch", 0, time_per_epoch)
    print_duration("Total Training", train_start_time, train_end_time)
    print("\nFinished model training.")

if __name__ == "__main__":
    main(sys.argv[1:])
