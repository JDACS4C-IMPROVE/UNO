import time
train_start_time = time.time()
import os
import sys
from pathlib import Path
from typing import Dict

# Script Dependencies: pandas, numpy, tensorflow

import numpy as np
import pandas as pd
import tensorflow as tf

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm

# [Req] Import params
from params import app_preproc_params, model_preproc_params, app_train_params, model_train_params

# Import custom made functions
from uno_utils_improve import (
    data_generator, 
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

# Model imports
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dropout, Lambda
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    LearningRateScheduler,
    EarlyStopping,
)

# Check tensorflow and GPU
print("Tensorflow Version:")
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

filepath = Path(__file__).resolve().parent  # [Req]

"""
# Notes: 
  - Model.fit initalizes batch before epoch, causing that generator to be off a batch size.
  - Do not use same generator to make predictions... results in index shift that cause v1=v2 error
  - Predictions are underestimates much more often than not... probably because there are lots of
    auc values close to 1 because we only have effective drugs in our dataset and sigmoid has small
    curvature, making extreme values very difficult. If we have lots of ineffective drugs, we will
    have extreme values close to 0 as well. Worth coming up with a more straightened out activation
    function to allow for extreme values more often.
  - power_yj scaler that is made from a different cross-study dataset can cause NaNs from exploding
    or vanishing gradients. This is because the power_yj scaler is not robust to extreme values and
    requires cleaning of the array before storing test scores.
  - Incorporate R2 calculation into the normal loss calculation? How it's done right now requires
    loading and predicting the entire dataset again, which is not efficient.
"""

# ---------------------
# [Req] Parameter lists
# ---------------------
preprocess_params = app_preproc_params + model_preproc_params
train_params = app_train_params + model_train_params

# ---------------------

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]


def read_architecture(params, hyperparam_space, arch_type):
    # Setup the architecture for cancer, drug, and interaction layers.
    """
    This function is made to allow for different hyperparameter spaces to be used for the architecture,
    allowing for global, block, and layer hyperparameter spaces when performing HPO. Depending on whether
    the hyperparameter space is global, block, or layer, the architecture will read from the global,
    block, or layer hyperparameters from the default model file for the cancer, drug, and interaction layers. 
    This way, the architecture can be defined in a more flexible way, allowing for more complex architectures 
    to be defined.
    """
    layers_size = []
    layers_dropout = []
    layers_activation = []

    if hyperparam_space == "global":
        if arch_type == "canc" or "drug":
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
    """Run model training.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set performance scores
    frm.create_outdir(outdir=params["model_outdir"])

    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])

    # ------------------------------------------------------
    # Reading hyperparameters
    # ------------------------------------------------------

    # Learning Hyperparams
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    generator_batch_size = params["generator_batch_size"]
    raw_max_lr = params["raw_max_lr"]
    raw_min_lr = raw_max_lr / 10000
    max_lr = raw_max_lr * batch_size
    min_lr = raw_min_lr * batch_size
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
    # Uses the read_architecture function to allow for different hyperparameter spaces
    canc_num_layers, canc_layers_size, canc_layers_dropout, canc_layers_activation = read_architecture(params, hyperparam_space, "canc")
    drug_num_layers, drug_layers_size, drug_layers_dropout, drug_layers_activation = read_architecture(params, hyperparam_space, "drug")
    interaction_num_layers, interaction_layers_size, interaction_layers_dropout, interaction_layers_activation = read_architecture(params, hyperparam_space, "interaction")

    # Final regression layer
    regression_activation = params["regression_activation"]

    # Print architecture in debug mode
    if train_debug:
        print("CANCER LAYERS:")
        print(canc_layers_size, canc_layers_dropout, canc_layers_activation)
        print("DRUG LAYERS:")
        print(drug_layers_size, drug_layers_dropout, drug_layers_activation)
        print("INTERACTION LAYERS:")
        print(
            interaction_layers_size,
            interaction_layers_dropout,
            interaction_layers_activation,
            )
        print("REGRESSION LAYER:")
        print(regression_activation)


    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    tr_data = pd.read_parquet(Path(params["train_ml_data_dir"])/train_data_fname)
    vl_data = pd.read_parquet(Path(params["val_ml_data_dir"])/val_data_fname)

    # Subset if setting true (for testing)
    if train_subset_data:
        # Define the total number of samples and the proportions for each stage
        total_num_samples = 5000
        stage_proportions = {"train": 0.8, "val": 0.1, "test": 0.1}   # should represent proportions given
        # Shuffle with num_samples set by total and stage
        tr_data = subset_data(tr_data, "Train", total_num_samples, stage_proportions)
        vl_data = subset_data(vl_data, "Validation", total_num_samples, stage_proportions)

    # Show data in debug mode
    if train_debug:
        print("TRAIN DATA:")
        print(tr_data.head())
        print(tr_data.shape)
        print("")
        print("VAL DATA:")
        print(vl_data.head())
        print(vl_data.shape)


    # Identify the Feature Sets from DataFrame
    num_ge_columns = len([col for col in tr_data.columns if col.startswith('ge')])
    num_md_columns = len([col for col in tr_data.columns if col.startswith('mordred')])

    # Separate input (features) and target and convert to numpy arrays
    # (better for tensorflow)
    x_train = tr_data.iloc[:, 1:].to_numpy()
    y_train = tr_data.iloc[:, 0].to_numpy()
    x_val = vl_data.iloc[:, 1:].to_numpy()
    y_val = vl_data.iloc[:, 0].to_numpy()

    # Slice the input tensor
    all_input = Input(shape=(num_ge_columns + num_md_columns,), name="all_input")
    canc_input = Lambda(lambda x: x[:, :num_ge_columns])(all_input)
    drug_input = Lambda(lambda x: x[:, num_ge_columns:num_ge_columns + num_md_columns])(all_input)

    # Cancer expression input and encoding layers
    canc_encoded = canc_input
    for i in range(canc_num_layers):
        canc_encoded = Dense(canc_layers_size[i], activation=canc_layers_activation[i])(canc_encoded)
        canc_encoded = Dropout(canc_layers_dropout[i])(canc_encoded)

    # Drug expression input and encoding layers
    drug_encoded = drug_input
    for i in range(drug_num_layers):
        drug_encoded = Dense(drug_layers_size[i], activation=drug_layers_activation[i])(drug_encoded)
        drug_encoded = Dropout(drug_layers_dropout[i])(drug_encoded)
    
    # Concatenated input and interaction layers
    interaction_input = Concatenate()([canc_encoded, drug_encoded])
    interaction_encoded = interaction_input
    for i in range(interaction_num_layers):
        interaction_encoded = Dense(
            interaction_layers_size[i], activation=interaction_layers_activation[i]
        )(interaction_encoded)
        interaction_encoded = Dropout(interaction_layers_dropout[i])(
            interaction_encoded
        )

    # Final output layer
    output = Dense(1, activation=regression_activation)(interaction_encoded)

    # Compile Model
    model = Model(inputs=all_input, outputs=output)
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
    )

    # Observe model if debugging mode
    if train_debug:
        model.summary()


    # Number of batches for data loading and callbacks
    # steps_per_epoch is for grad descent batches while train_steps is for r2_train
    steps_per_epoch = int(np.ceil(len(x_train) / batch_size))
    validation_steps = int(np.ceil(len(x_val) / generator_batch_size))


    # Instantiate callbacks

    # Learning rate scheduler
    lr_scheduler = LearningRateScheduler(
        lambda epoch: warmup_scheduler(
            epoch, model.optimizer.lr, warmup_epochs, initial_lr, max_lr, warmup_type
        )
    )

    # Reduce learing rate
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    )


    # R2 (efficient or accurate)

    # Calculate ss_tot for r2
    train_ss_tot = calculate_sstot(y_train)
    val_ss_tot = calculate_sstot(y_val)

    # Efficient (calculated through epoch) (averaging errors with smaller last batch)
    r2_callback= R2Callback_efficient(
        train_ss_tot=train_ss_tot,
        val_ss_tot=val_ss_tot
    )


    epoch_start_time = time.time()


    # Make separate generators for training (fixing index issue)
    train_gen = data_generator(x_train, y_train, batch_size, shuffle=True, peek=True, verbose=False)
    val_gen = data_generator(x_val, y_val, generator_batch_size, shuffle=False, verbose=False)

    # Fit model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[r2_callback, lr_scheduler, reduce_lr, early_stopping],
    )


    # Calculate the time per epoch
    epoch_end_time = time.time()
    total_epochs = len(history.history['loss'])  # Get the actual number of epochs
    global time_per_epoch 
    time_per_epoch = (epoch_end_time - epoch_start_time) / total_epochs

    # Save model
    model.save(modelpath)


    # Batch prediction (and flatten inside function)
    # Make sure to make new generator state so no index problem
    val_pred, val_true = batch_predict(model, data_generator(x_val, y_val, generator_batch_size), validation_steps)
    

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    # Data must be subsetted in preprocess AND train or neither. Dangerous if parameter changes without running again
    if (train_subset_data and preprocess_subset_data) or (not train_subset_data and not preprocess_subset_data):    
        frm.store_predictions_df(
            params,
            y_true=val_true,
            y_pred=val_pred,
            stage="val",
            outdir=params["model_outdir"],
        )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true,
        y_pred=val_pred,
        stage="val",
        outdir=params["model_outdir"],
        metrics=metrics_list,
    )


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params
    params = frm.initialize_parameters(
        filepath,
        default_model="uno_default_model.txt",
        additional_definitions=additional_definitions,
        # required=req_train_params,
        required=None,
    )
    run(params)
    train_end_time = time.time()
    print_duration("One epoch", 0, time_per_epoch)
    print_duration("Total Training", train_start_time, train_end_time)
    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
