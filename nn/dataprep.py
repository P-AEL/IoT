import pandas as pd, numpy as np, torch
from copy import deepcopy


def cutoff_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    args:   df: pd.DataFrame
            start_date: str
            end_date: str

    returns: pd.DataFrame
    """
    df=deepcopy(df)
    return df[df["date_time"].between(start_date, end_date)]

def build_lvl_df(df: pd.DataFrame, device_ids: list, output_cols: list, reset_ind: bool=True) -> pd.DataFrame:
    """
    args:   df: pd.DataFrame
            device_ids: list
            output_cols: list
            reset_ind: bool

    returns: pd.DataFrame
    """
    df=deepcopy(df)
    return df[df["device_id"].isin(device_ids)][["date_time"]+output_cols].groupby("date_time").agg("mean").reset_index(drop=True) if reset_ind else df[df["device_id"].isin(device_ids)][["date_time"]+output_cols].groupby("date_time").agg("mean")

def train_test_split(X: np.array, y: np.array, train_size: float= 0.985) -> np.array:
    """
    args:   
        X: np.array: features
        y: np.array: target
        train_size: float: size of the training set
    
    returns:
        X_train: np.array: training features
        X_test: np.array: testing features
        y_train: np.array: training target
        y_test: np.array: testing target
    """
    split_index=int(len(X) * train_size)
    X_train=torch.tensor(X[:split_index], dtype=torch.float32)
    X_test=torch.tensor(X[split_index:], dtype=torch.float32)
    y_train=torch.tensor(y[:split_index], dtype=torch.float32)
    y_test=torch.tensor(y[split_index:], dtype=torch.float32)
    return X_train, X_test, y_train, y_test

def format_tensor(X: torch.tensor, window_size: int= 48) -> torch.tensor:
    """
    args:   X: torch.tensor
            window_size: int
    
    returns: torch.tensor        
    """
    X_new = torch.stack([torch.cat((torch.zeros(window_size-i, X.shape[1]), X[:i])) if i < window_size else X[i-window_size:i] for i in range(0, len(X))])

    return X_new[1:]