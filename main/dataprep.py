import pandas as pd, numpy as np, plotly.graph_objects as go, plotly.express as px
from copy import deepcopy
from scipy import stats
import torch

def group_data(df: pd.DataFrame, freq: str="h") -> pd.DataFrame:
    """
    args:   df: pd.DataFrame
            freq: str

    returns: pd.DataFrame
    """
    df = deepcopy(df)
    df['date_time'] = pd.to_datetime(df['date_time'])
    all_columns_float_type = df.select_dtypes(include=['float64']).columns
    all_columns_float_type = all_columns_float_type.tolist()
    all_columns_float_type.append('device_id')
    all_columns_float_type.append('date_time')
    all_columns_int_type = df.select_dtypes(include=['int']).columns   
    all_columns_int_type = all_columns_int_type.tolist()
    all_columns_int_type.append('device_id')
    all_columns_int_type.append('date_time')

    df_hourly_float_values = df[all_columns_float_type].groupby(['device_id', pd.Grouper(key='date_time', freq=freq)]).agg("mean").reset_index()
    df_hourly_int_values = df[all_columns_int_type].groupby(['device_id', pd.Grouper(key='date_time', freq=freq)]).agg("max").reset_index()

    merged_df = df_hourly_float_values.merge(df_hourly_int_values, on=['device_id', 'date_time'])
    return merged_df

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

def plt_fig(df: pd.DataFrame, y: str="tmp", mode: str="lines+markers", trendline: bool=False):
    """
    args:  df: pd.DataFrame
            y: str
            mode: str
            trendline: bool

    returns: go.Figure
    """
    if trendline:
        fig = px.scatter(df, x="date_time", y=y, trendline="rolling",trendline_color_override="red", trendline_options=dict(window=240, win_type="gaussian", function_args=dict(std=2)))
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df["date_time"], y=df[y], mode=mode)
        )
    fig.update_layout(xaxis_title= "Time", yaxis_title= y)
    return fig

    