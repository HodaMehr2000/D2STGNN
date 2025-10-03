# Load-data

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import pickle
import numpy as np
from functools import partial

from dataloader import DataLoader
from utils.cal_adj import (
    calculate_scaled_laplacian,
    calculate_symmetric_normalized_laplacian,
    symmetric_message_passing_adj,
    transition_matrix,
)

# ----------------------------
# Normalization utilities
# ----------------------------

def re_normalization(x, mean, std):
    """
    Standard re-normalization
    """
    return x * std + mean


def max_min_normalization(x, _max, _min, eps=1e-8):
    """
    Max-min normalization to [-1, 1].
    Supports scalar or array (broadcastable) _max/_min.
    eps avoids division-by-zero when _max == _min.
    """
    x = 1.0 * (x - _min) / (np.maximum(_max - _min, eps))
    x = x * 2.0 - 1.0
    return x


def re_max_min_normalization(x, _max, _min):
    """
    Max-min re-normalization from [-1, 1] back to original scale.
    Supports scalar or array (broadcastable) _max/_min.
    """
    x = (x + 1.0) / 2.0
    x = 1.0 * x * (_max - _min) + _min
    return x


class StandardScaler:
    """
    Standardize the input with mean/std.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# ----------------------------
# IO helpers
# ----------------------------

def load_pickle(pickle_file):
    """
    Load pickle data (with latin1 fallback).
    """
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')


def _y_to_num_N_pred(y):
    """
    Convert y to shape [num, N, pred_len] robustly.
    Accepts:
      - [num, pred_len, N, C]  (C can be 1 or >1)
      - [num, pred_len, N]
      - [num, N, pred_len]
    Strategy:
      - If 4D: transpose to [num, N, pred_len, C]; if C>1, take channel 0.
      - If 3D: heuristics — if axis1 looks like pred_len (6,12,24,36), transpose (0,2,1)
               else assume already [num, N, pred_len].
    """
    if y.ndim == 4:
        # [num, pred_len, N, C] -> [num, N, pred_len, C]
        y_t = np.transpose(y, (0, 2, 1, 3))
        return y_t[..., 0] if y_t.shape[-1] > 1 else np.squeeze(y_t, axis=-1)  # [num, N, pred_len]
    elif y.ndim == 3:
        # could be [num, pred_len, N] or [num, N, pred_len]
        if y.shape[1] in (6, 12, 24, 36):
            return np.transpose(y, (0, 2, 1))  # [num, N, pred_len]
        return y  # assume already [num, N, pred_len]
    else:
        raise ValueError(f"Unexpected y ndim: {y.ndim}")


# ----------------------------
# Dataset loading
# ----------------------------

def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    """
    Load the whole datasets.

    Args:
        data_dir: str, e.g., 'datasets/PEMS08'
    Returns:
        dict with x_/y_ splits, loaders, and scaler
    """
    data_dict = {}

    # Read processed .npz files: train/val/test
    for mode in ['train', 'val', 'test']:
        npz_path = os.path.join(data_dir, mode + '.npz')
        _ = np.load(npz_path, allow_pickle=True)
        data_dict['x_' + mode] = _['x']
        data_dict['y_' + mode] = _['y']

    # Traffic flow datasets -> min-max to [-1,1]
    if dataset_name in ('PEMS04', 'PEMS08'):
        # Robust target shaping to [num, N, pred_len]
        y_train = _y_to_num_N_pred(data_dict['y_train'])
        y_val   = _y_to_num_N_pred(data_dict['y_val'])
        y_test  = _y_to_num_N_pred(data_dict['y_test'])

        # Try load per-node/per-step min/max next to data_dir; fallback to scalar min/max from y_train
        min_path = os.path.join(data_dir, "min.pkl")
        max_path = os.path.join(data_dir, "max.pkl")
        has_minmax = os.path.exists(min_path) and os.path.exists(max_path)

        _min_arr = _max_arr = None
        _min_scalar = _max_scalar = None

        if has_minmax:
            _min_loaded = load_pickle(min_path)
            _max_loaded = load_pickle(max_path)
            if isinstance(_min_loaded, np.ndarray) and isinstance(_max_loaded, np.ndarray):
                _min_arr, _max_arr = _min_loaded, _max_loaded
            else:
                _min_scalar, _max_scalar = float(_min_loaded), float(_max_loaded)
        else:
            _min_scalar = float(np.min(y_train))
            _max_scalar = float(np.max(y_train))
            # Try to persist for later runs (non-fatal if fails)
            try:
                with open(min_path, 'wb') as f: pickle.dump(_min_scalar, f)
                with open(max_path, 'wb') as f: pickle.dump(_max_scalar, f)
            except Exception as _e:
                print("[WARN] Could not write min/max pkl:", _e)

        # Normalize to [-1,1]
        if _min_arr is not None and _max_arr is not None:
            # Expect broadcastable shapes. If [N,1], expand to [N,pred_len]
            mm_min, mm_max = _min_arr, _max_arr
            if mm_min.ndim == 2 and mm_min.shape[1] == 1:
                mm_min = np.repeat(mm_min, y_train.shape[-1], axis=1)
                mm_max = np.repeat(mm_max, y_train.shape[-1], axis=1)
            y_train_new = max_min_normalization(y_train, mm_max, mm_min)
            y_val_new   = max_min_normalization(y_val,   mm_max, mm_min)
            y_test_new  = max_min_normalization(y_test,  mm_max, mm_min)
            inv_scaler  = partial(re_max_min_normalization, _max=mm_max, _min=mm_min)  # << no lambda
        else:
            y_train_new = max_min_normalization(y_train, _max_scalar, _min_scalar)
            y_val_new   = max_min_normalization(y_val,   _max_scalar, _min_scalar)
            y_test_new  = max_min_normalization(y_test,  _max_scalar, _min_scalar)
            inv_scaler  = partial(re_max_min_normalization, _max=_max_scalar, _min=_min_scalar)  # << no lambda

        # Back to model's expected shape: [num, pred_len, N]
        data_dict['y_train'] = np.transpose(y_train_new, (0, 2, 1))
        data_dict['y_val']   = np.transpose(y_val_new,   (0, 2, 1))
        data_dict['y_test']  = np.transpose(y_test_new,  (0, 2, 1))

        # Build DataLoaders and scaler (inverse)
        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader']   = DataLoader(data_dict['x_val'],   data_dict['y_val'],   valid_batch_size)
        data_dict['test_loader']  = DataLoader(data_dict['x_test'],  data_dict['y_test'],  test_batch_size)
        data_dict['scaler']       = inv_scaler  # picklable now

    else:
        # Traffic speed datasets -> z-score using train stats
        scaler = StandardScaler(
            mean=data_dict['x_train'][..., 0].mean(),
            std =data_dict['x_train'][..., 0].std()
        )
        for mode in ['train', 'val', 'test']:
            data_dict['x_' + mode][..., 0] = scaler.transform(data_dict['x_' + mode][..., 0])
            data_dict['y_' + mode][..., 0] = scaler.transform(data_dict['y_' + mode][..., 0])

        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader']   = DataLoader(data_dict['x_val'],   data_dict['y_val'],   valid_batch_size)
        data_dict['test_loader']  = DataLoader(data_dict['x_test'],  data_dict['y_test'],  test_batch_size)
        data_dict['scaler']       = scaler

    return data_dict


# ----------------------------
# Adjacency loading
# ----------------------------

def load_adj(file_path, adj_type):
    """
    Load adjacency matrix and preprocess it.
    """
    try:
        # METR and PEMS_BAY
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
    except Exception:
        # PEMS04 provides a plain adj matrix pickle
        adj_mx = load_pickle(file_path)

    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32).todense()]
    elif adj_type == "original":
        adj = adj_mx
    else:
        assert False, "adj type not defined"

    return adj, adj_mx



