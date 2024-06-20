import pandas as pd, numpy as np, plotly.graph_objects as go, plotly.express as px
from copy import deepcopy
import torch
import torch.utils
import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import namedtuple
import streamlit as st
logging.basicConfig(level=logging.INFO)
import influxdb_client
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS
import pathlib



def check_file_exists(filepath: str):
    """
    Check if a file exists.
    """
    if not os.path.exists(filepath):
        logging.error(f"File {filepath} does not exist.")
        raise FileNotFoundError(f"File {filepath} does not exist.")


def combine_files(file_paths):
    """
    Combine data from multiple files into a single DataFrame.
    """
    combined_df = pd.DataFrame()

    for file_path in file_paths:
        check_file_exists(file_path)
        try:
            df = pd.read_csv(file_path, sep=';', header=1)
            combined_df = pd.concat([combined_df, df])
        except Exception as e:
            logging.error(f"Failed to read file {file_path}: {e}")
            continue

    return combined_df


def save_to_parquet(df, filepath):
    """
    Save a DataFrame to a Parquet file.
    """
    try:
        df.to_parquet(filepath)
        logging.info(f"Data saved to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save data to {filepath}: {e}")


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
        fig = px.scatter(df, x= "date_time", y=y, trendline="rolling", trendline_color_override="red", trendline_options=dict(window=24, win_type="gaussian", function_args=dict(std=2)))
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df["date_time"], y=df[y], mode=mode)
        )
    fig.update_layout(xaxis_title= "Time", yaxis_title= y)
    return fig


DataLoaders = namedtuple('DataLoaders', ['train', 'test'])
Data = namedtuple('Data', ['x', 'y', 'scaler', 'loader'])

def update_Data(obs: pd.DataFrame, window_size: int, target: str, features: list) -> Data:
    """
    Update the Data namedtuple.

    args:   obs: pd.DataFrame
            window_size: int
            target: str
            features: list
    returns: Data
    """
    x = []
    y = []
    seq_size = window_size
    for g_id in obs['group'].unique():
        group_df = obs[obs['group'] == g_id]
        for i in range(len(group_df) - seq_size):
            window = group_df[i:(i + seq_size)]
            x.append(window[features].values)
        y.extend(group_df[f"{target}"].iloc[seq_size:])

    x, y = torch.tensor(np.array(x), dtype= torch.float32).view(-1, seq_size, 4), torch.tensor(y, dtype= torch.float32).view(-1, 1)

    return Data(x, y, None, None)

def create_DataLoader(filepath: str="agg_hourly.parquet", window_size: int=50, train_ratio: float=0.8, batch_size: int=64, target: str="tmp", features: list=["CO2", "hum", "VOC", "tmp"], scaling: bool=True) -> dict:
    """
    Create a DataLoader for training and testing data.

    args:   filepath: str
            window_size: int
            train_ratio: float
            batch_size: int
            target: str
            features: list
            scaling: bool

    returns: dict
    """
    check_file_exists(filepath)  
    data = pd.read_parquet(filepath)
    df = deepcopy(data)

    df.date_time = pd.to_datetime(df.date_time)
    df = df[['device_id', 'date_time'] + features]
    timedelta = 3600
    df['consecutive_data_point'] = (df['date_time'] - df['date_time'].shift(1)).dt.total_seconds() == timedelta
    df['consecutive_data_point'] = df['consecutive_data_point'].astype(int)
    df['reset'] = (df['consecutive_data_point'] == 0) | (df['device_id'] != df['device_id'].shift(1))
    df['group'] = df['reset'].cumsum()
    df['consecutive_data_points'] = df.groupby(['device_id', 'group'])['consecutive_data_point'].cumsum() - df['consecutive_data_point']
    df['group_size'] = df.groupby(['device_id', 'group'])['consecutive_data_point'].transform('count')
    df_cpy = deepcopy(df[df['group_size'] > window_size])
    df_cpy.drop(['reset', 'consecutive_data_point', 'consecutive_data_points', 'group_size'], axis=1, inplace=True)
    threshold_date = df_cpy.sort_values('date_time', ascending=True)['date_time'].quantile(train_ratio)
    df_train = df_cpy[df_cpy['date_time'] < threshold_date]
    df_test = df_cpy[df_cpy['date_time'] >= threshold_date]

    scaler = None
    if scaling:
        scaler = StandardScaler()
        df_train_scaled, df_test_scaled = deepcopy(df_train), deepcopy(df_test)
        df_train_scaled[features] = scaler.fit_transform(df_train_scaled[features])
        df_test_scaled[features] = scaler.transform(df_test_scaled[features])
        df_train = df_train_scaled
        df_test = df_test_scaled

    train_data = update_Data(df_train, window_size, target, features)
    test_data = update_Data(df_test, window_size, target, features)

    train_dataset = TensorDataset(train_data.x, train_data.y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_dataset = TensorDataset(test_data.x, test_data.y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    train_data = train_data._replace(loader=train_loader, scaler=scaler)
    test_data = test_data._replace(loader=test_loader, scaler=scaler)

    return {'train': train_data, 'test': test_data}


# streamlit functions
@st.cache_data
def load_data(filename: str = "agg_hourly.parquet",use_influx_db: bool = False) -> pd.DataFrame:
    if not use_influx_db:
        filepath = os.path.join("./data/aggregated_data/", filename)
        if not os.path.exists(filepath):
            logging.error(f"File {filepath} does not exist.")
            raise FileNotFoundError(f"File {filepath} does not exist.")
        df = pd.read_parquet(filepath)
    else:
        # Secrets
        token = open("token.txt", "r").read()
        url = "https://iwi-i-influx-db-01.hs-karlsruhe.de:8086"
        bucket = "iot_gebaeude_a"
        org = "Vorlesung"

        # Client erstellen und Read/Write API anlegen
        write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
        query_api = write_client.query_api()
        query = f"""from(bucket: "{bucket}")
        |> range(start: 2021-03-17T23:30:00Z)"""

        tables = query_api.query(query, org=org)

        data = []
        for table in tables:
            for record in table.records:
                # Extract the necessary fields
                time = record.get_time()
                value = record.get_value()
                measurement = record.values.get('_measurement')
                field = record.values.get('_field')
                sensor = record.values.get('sensor')
                
                # Append to data list
                data.append([time, value, measurement, field, sensor])

        # Define column names
        column_names = ['date_time', 'value', 'measurement', 'field', 'sensor']

        # Create the DataFrame
        df = pd.DataFrame(data, columns=column_names)
        df_new = []
        for time in df['date_time'].unique():
            df_time = df.loc[df['date_time'] == time]
            tmp = df_time[df_time['field'] == 'tmp']['value'].values[0]
            CO2 = df_time[df_time['field'] == 'CO2']['value'].values[0]
            BLE = df_time[df_time['field'] == 'BLE']['value'].values[0]
            IR = df_time[df_time['field'] == 'IR']['value'].values[0]
            WIFI = df_time[df_time['field'] == 'WIFI']['value'].values[0]
            VOC = df_time[df_time['field'] == 'VOC']['value'].values[0]
            bandwidth = df_time[df_time['field'] == 'bandwidth']['value'].values[0]
            channel_index = df_time[df_time['field'] == 'channel_index']['value'].values[0]
            channel_rssi = df_time[df_time['field'] == 'channel_rssi']['value'].values[0]
            f_cnt = df_time[df_time['field'] == 'f_cnt']['value'].values[0]
            hum = df_time[df_time['field'] == 'hum']['value'].values[0]
            rssi = df_time[df_time['field'] == 'rssi']['value'].values[0]
            spreading_factor = df_time[df_time['field'] == 'spreading_factor']['value'].values[0]
            vis = df_time[df_time['field'] == 'vis']['value'].values[0]
            device_id = df_time[df_time['field'] == 'device_id']['value'].values[0]
            snr = df_time[df_time['field'] == 'snr']['value'].values[0]
            gateway = df_time[df_time['field'] == 'gateway']['value'].values[0]
            df_new.append([time, device_id, tmp, hum, CO2, VOC, vis, IR, WIFI, BLE, rssi, channel_rssi, snr, gateway, channel_index, spreading_factor, bandwidth, f_cnt])
        df_new = pd.DataFrame(df_new, columns=['date_time', 'device_id', 'tmp', 'hum', 'CO2', 'VOC', 'vis', 'IR', 'WIFI', 'BLE', 'rssi', 'channel_rssi', 'snr', 'gateway', 'channel_index', 'spreading_factor', 'bandwidth', 'f_cnt'])
        if filename == "agg_hourly.parquet":
            df_new = group_data(df_new, freq="h")
    return df_new

