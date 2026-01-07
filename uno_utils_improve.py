import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Union
from tensorflow.keras.callbacks import (
    Callback,
)


# ------------------------------------------------------
# Preprocess Utils
# ------------------------------------------------------


def get_common_samples(
    canc_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    rsp_df: pd.DataFrame,
    canc_col_name: str,
    drug_col_name: str,
):
    """
    Args:
        canc_df (pd.Dataframe): cell features df.
        drug_df (pd.Dataframe): drug features df.
        rsp_df (pd.Dataframe): drug response df.
        canc_col_name (str): Column name that contains the cancer sample ids.
        drug_col_name (str): Column name that contains the drug ids.

    Returns:
        Cancer, drug, and response dataframes with only the common samples 
        between them all.

    Justification:
        When creating scalers, it's important to create on only the drugs/cell
        lines present in that dataset. Also, filtering unnecessary data before
        merging saves memory and computation time when later merging
    """
    # Filter response according to all
    rsp_df = rsp_df.merge(
       canc_df[canc_col_name], on=canc_col_name, how="inner"
    )
    rsp_df = rsp_df.merge(
       drug_df[drug_col_name], on=drug_col_name, how="inner"
    )
    # Filter all according to response
    canc_df = canc_df[
        canc_df[canc_col_name].isin(rsp_df[canc_col_name])
    ].reset_index(drop=True)
    drug_df = drug_df[
        drug_df[drug_col_name].isin(rsp_df[drug_col_name])
    ].reset_index(drop=True)

    return canc_df, drug_df, rsp_df



# ------------------------------------------------------
# Train Utils
# ------------------------------------------------------

def get_optimizer(optimizer_name, initial_lr):
    if optimizer_name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=initial_lr)
    elif optimizer_name == "SGD":
        return tf.keras.optimizers.SGD(learning_rate=initial_lr)
    elif optimizer_name == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=initial_lr)
    elif optimizer_name == "Adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=initial_lr)
    elif optimizer_name == "Adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate=initial_lr)
    elif optimizer_name == "Adamax":
        return tf.keras.optimizers.Adamax(learning_rate=initial_lr)
    elif optimizer_name == "Nadam":
        return tf.keras.optimizers.Nadam(learning_rate=initial_lr)
    elif optimizer_name == "Ftrl":
        return tf.keras.optimizers.Ftrl(learning_rate=initial_lr)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not recognized")


def get_column_ranges(file_path):
    # Read only the header of the CSV file
    header = pd.read_csv(file_path, nrows=0)  # nrows=0 means no data rows are read

    # Extract column names from the header
    columns = header.columns

    # Initialize indices
    ge_start_index = ge_end_index = md_start_index = md_end_index = None
    ge_num = md_num = 0

    # Loop through columns to find start and end indices for 'ge' and 'mordred' columns
    for i, col in enumerate(columns):
        if col.startswith('ge'):
            ge_num += 1
            ge_end_index = i  # Update last seen 'ge' column index
            if ge_start_index is None:
                ge_start_index = i  # Set start index at first occurrence
            
        elif col.startswith('mordred'):
            md_num += 1
            md_end_index = i  # Update last seen 'mordred' column index
            if md_start_index is None:
                md_start_index = i  # Set start index at first occurrence


def calculate_sstot(y):
    """
    Calculate the total sum of squares (SStot) using a NumPy array.

    :param y: NumPy array of observed values.
    :return: Total sum of squares (SStot).
    """
    # Calculate the mean of the observed values
    y_mean = np.mean(y)

    # Calculate the mean sum of squares total
    # (mean just because of how metric averaging works in tensorflow)
    ss_tot = np.mean((y - y_mean) ** 2)

    return ss_tot


class R2Callback_efficient(Callback):
    def __init__(self, train_ss_tot, val_ss_tot):
        super().__init__()
        self.train_ss_tot = train_ss_tot
        self.val_ss_tot = val_ss_tot

    def on_epoch_end(self, epoch, logs=None):
        # Calculate R2
        train_ss_res = logs.get("loss")
        val_ss_res = logs.get("val_loss")
        train_r2 = 1 - (train_ss_res / self.train_ss_tot)
        val_r2 = 1 - (val_ss_res / self.val_ss_tot)
        # Enter R2 into logs
        logs["r2_train"] = train_r2
        logs["r2_val"] = val_r2
        # Print
        # print(f'\nEpoch: {epoch + 1}, Train R2: {train_r2}, Val R2: {val_r2} \n')


class R2Callback_accurate(Callback):
    def __init__(self, model, r2_train_generator, r2_val_generator, train_steps, validation_steps, train_ss_tot, val_ss_tot):
        super().__init__()
        self.model = model
        self.train_generator = r2_train_generator
        self.val_generator = r2_val_generator
        self.train_steps = train_steps
        self.validation_steps = validation_steps
        self.train_ss_tot = train_ss_tot
        self.val_ss_tot = val_ss_tot

    def on_epoch_end(self, epoch, logs=None):
        # print("R2 Start")
        # Calculate R2
        train_r2 = self.compute_r2(self.train_generator, self.train_steps, self.train_ss_tot)
        val_r2 = self.compute_r2(self.val_generator, self.validation_steps, self.val_ss_tot)
        # Enter R2 into logs
        logs["r2_train"] = train_r2
        logs["r2_val"] = val_r2
        # Print
        print(f'\nEpoch: {epoch + 1}, Train R2: {train_r2}, Val R2: {val_r2} \n')
        # print("R2 End")

    def compute_r2(self, data_generator, steps, mean_ss_tot):
        # Get true and predicted values
        y_pred, y_true = batch_predict(self.model, data_generator, steps)

        # Compute R2
        mean_ss_res = np.mean((y_true - y_pred) ** 2)
        r2 = 1 - (mean_ss_res / mean_ss_tot)

        return r2


def warmup_scheduler(epoch, lr, warmup_epochs, initial_lr, max_lr, warmup_type):
    if epoch <= warmup_epochs:
        if warmup_type == "none" or warmup_type == "constant":
            lr = max_lr
        elif warmup_type == "linear":
            lr = initial_lr + (max_lr - initial_lr) * epoch / warmup_epochs
        elif warmup_type == "quadratic":
            lr = initial_lr + (max_lr - initial_lr) * ((epoch / warmup_epochs) ** 2)
        elif warmup_type == "exponential":
            lr = initial_lr * ((max_lr / initial_lr) ** (epoch / warmup_epochs))
        else:
            raise ValueError("Invalid warmup type")
    return float(lr)  # Ensure returning a float value


# ------------------------------------------------------
# Shared Utils
# ------------------------------------------------------


def subset_data(rsp: pd.DataFrame, stage: str, total_num_samples: int, stage_proportions: Dict):
    # Check for valid stage
    if stage not in stage_proportions:
        raise ValueError(f"Unrecognized stage when subsetting data: {stage}")
    # Check for small datasets
    naive_num_samples = int(total_num_samples * stage_proportions[stage])
    stage_num_samples = min(naive_num_samples, rsp.shape[0])
    # Print info
    if naive_num_samples >= rsp.shape[0]:
        print(f"Small {stage.capitalize()} Dataset of Size {stage_num_samples}. "
        f"Subsetting to {naive_num_samples} Is Skipped")
    else:
        print(f"Subsetting {stage} Data To: {stage_num_samples}")
    # Subset data
    rsp = rsp.sample(n=stage_num_samples).reset_index(drop=True)

    return rsp
    

def merge(rsp, ge, md, indices, params, preserve_order=False, debug=False):
    if preserve_order:
        # Add an 'order' column to 'rsp' to keep track of the original order... 
        # sometimes merging messes up the order when the 'on' column is not unique, which matters for comparing to the original rsp
        rsp_with_order = rsp.iloc[indices].copy()
        rsp_with_order['order'] = range(len(rsp_with_order))
        rsp_for_merge = rsp_with_order
    else:
        rsp_for_merge = rsp.iloc[indices]

    # Perform merging operations
    merged_df = rsp_for_merge.merge(ge, on=params["canc_col_name"], how="inner")
    merged_df = merged_df.merge(md, on=params["drug_col_name"], how="inner")

    # Drop the columns used for merging, if necessary
    merged_df.drop([params["canc_col_name"], params["drug_col_name"]], axis=1, inplace=True)

    if preserve_order:
        # Sort the merged DataFrame based on the 'order' column to restore the original order
        merged_df.sort_values(by='order', inplace=True)
        # Drop the 'order' column as it's no longer needed
        merged_df.drop(['order'], axis=1, inplace=True)

    # Show dataframes if on debug mode
    if debug:
        print("Merged Data:")
        print(merged_df.head())
        print("")

    # Extract train and target data
    x_data = merged_df.drop(params["y_col_name"], axis=1).values
    y_data = merged_df[params["y_col_name"]].values

    return x_data, y_data


def data_merge_generator(rsp, ge, md, batch_size, params, shuffle=False, peek=False, merge_preserve_order=False, verbose=False):
    num_samples = len(rsp)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)

    while True:    # Loop indefinitely for epochs

        if peek:    # Give first batch unshuffled and don't change start index when peeking for training
            end = min(batch_size, num_samples)
            if verbose:
                print(f"Generating peeking batch up to index {end}")
            batch_indices = indices[:end]
            batch_x, batch_y = merge(rsp, ge, md, batch_indices, params, preserve_order=merge_preserve_order, debug=params['train_debug'])
            peek = False
            yield (batch_x, batch_y)

        # Shuffle indices at the start of each epoch after the peek, if shuffle is enabled
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            # Print batch indices if verbose
            if verbose:
                # Warning: calling verbose when shuffling will usually clutter output
                if shuffle:
                    if len(batch_indices) <= 16:
                        print(f"Batch indices: {np.sort(batch_indices)}")
                    else:
                        print(f"Printing batch indices would clutter output. Skipped.")
                    print(f"Length: {len(batch_indices)}")
                else:
                    print(f"Generating batch from index {start} to {end}")
                # traceback.print_stack()

            # Generate batches
            batch_x, batch_y = merge(rsp, ge, md, batch_indices, params, preserve_order=merge_preserve_order)

            # Yield the current batch
            yield (batch_x, batch_y)


def data_generator(x_data, y_data, batch_size, shuffle=False, peek=False, verbose=False):
    num_samples = len(x_data)
    indices = np.arange(num_samples)
    
    if peek:    # Give first batch unshuffled and don't change start index when peeking for training
        end = min(batch_size, num_samples)
        if verbose:
            print(f"Generating peeking batch up to index {end}")
        batch_x = x_data[:end]
        batch_y = y_data[:end]
        peek = False
        yield (batch_x, batch_y)

    while True:    # Loop indefinitely for epochs
        # Shuffle indices at the start of each epoch after the peek, if shuffle is enabled
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            # Print batch indices if verbose
            if verbose:
                # Warning: calling verbose when shuffling will usually clutter output
                if shuffle:
                    if len(batch_indices) < 16:
                        print(f"Batch indices: {np.sort(batch_indices)}")
                    else:
                        print(f"Printing batch indices would clutter output. Skipped.")
                    print(f"Length: {len(batch_indices)}")
                else:
                    print(f"Generating batch from index {start} to {end}")
                # traceback.print_stack()

            # Generate batches
            batch_x = x_data[batch_indices]
            batch_y = y_data[batch_indices]

            # Yield the current batch
            yield (batch_x, batch_y)



def batch_predict(model, data_generator, steps, flatten=True, verbose=False):
    predictions = []
    true_values = []
    for _ in range(steps):
        # print("Batch Predict get next")
        x, y = next(data_generator)
        pred = model.predict(x, verbose=0)
        if flatten:
            pred = pred.flatten()
            y = y.flatten()
        predictions.extend(pred)
        true_values.extend(y)
        if verbose:
            print("Batch Predict:")
            print(f"Predictions: {len(predictions)}")
            print(f"True: {len(true_values)}")
    return np.array(predictions), np.array(true_values)


# ------------------------------------------------------
# General Utils
# ------------------------------------------------------

def print_duration(activity: str, start_time: float, end_time: float):
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    print(f"Time for {activity}: {hours} hours, {minutes} minutes, and {seconds} seconds\n")


def clean_arrays(pred, true):
    # Initialize clean arrays
    pred_clean = pred
    true_clean = true

    # Find NaN indices and remove
    nan_indices = np.where(np.isnan(pred))[0]
    pred_clean = np.delete(pred_clean, nan_indices)
    true_clean = np.delete(true_clean, nan_indices)

    # Find infinity indices and remove
    inf_indices = np.where(np.isinf(pred))[0]
    pred_clean = np.delete(pred_clean, inf_indices)
    true_clean = np.delete(true_clean, inf_indices)

    # Print the number and percent of removed indices
    start_len = len(pred)
    end_len = len(pred_clean)
    print(f"Removed {start_len - end_len} values due to NaN or infinity values.")
    print(f"Removed {100 * (start_len - end_len) / start_len:.3f}% of data due to NaN or infinity values.")

    return pred_clean, true_clean


def check_array(array):
    # Print shape
    print(f"Shape: {array.shape}")

    # Print the first few values
    print("First few values:", array[:5])

    # Check and print indices/values for NaN values
    nan_indices = np.where(np.isnan(array))[0]
    print("Indices of NaN:", nan_indices[:5])

    # Check and print indices/values for infinity values
    inf_indices = np.where(np.isinf(array))[0]
    print("Indices of infinity:", inf_indices[:5])



# ------------------------------------------------------
# Not Yet Used Utils
# ------------------------------------------------------

# TO-DO related to lincs
def gene_selection(df: pd.DataFrame, genes_fpath: Union[Path, str], canc_col_name: str):
    """Takes a dataframe omics data (e.g., gene expression) and retains only
    the genes specified in genes_fpath.
    """
    with open(genes_fpath) as f:
        genes = [str(line.rstrip()) for line in f]
    # genes = ["ge_" + str(g) for g in genes]  # This is for our legacy data
    # print("Genes count: {}".format(len(set(genes).intersection(set(df.columns[1:])))))
    genes = list(set(genes).intersection(set(df.columns[1:])))
    # genes = drp.common_elements(genes, df.columns[1:])
    cols = [canc_col_name] + genes
    return df[cols]