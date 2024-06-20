import streamlit as st, pandas as pd, plotly.graph_objects as go, dataprep as dp, torch.nn as nn, numpy as np
import torch
import foo
import logging
import os
import json
from datetime import datetime, date, time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
logging.basicConfig(level=logging.INFO)
import influxdb_client
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS


# Page config
st.set_page_config(
    page_title="Sensor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Functions
def read_credentials(filename: str= "credentials.txt"):
    """
    Reads the credentials from a given filename.

    args:   filename: str
    returns: dict
    """
    with open(filename, "r") as file:
        credentials = json.load(file)
    return credentials


def extract_data_from_influxdb(credentials, query):
    # Client erstellen und Read/Write API anlegen
    write_client = influxdb_client.InfluxDBClient(url= credentials['url'], token= credentials['token'], org= credentials['org'])
    query_api = write_client.query_api()
    tables = query_api.query(query, org= credentials['org'])

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
    return data

@st.cache_data
def load_data(filename: str= "agg_hourly.parquet", use_influx_db: bool= False) -> pd.DataFrame:
    """
    Loads data from given filename or InfluxDB.
    
    args:   filename: str
            use_influx_db: bool
    returns pd.DataFrame
    """
    df_new = []

    if use_influx_db:
        # Get credentials for InfluxDB
        credentials = read_credentials("credentials.json")
        query = f"""from(bucket: "{credentials['bucket']}")
        |> range(start: 2021-03-17T23:30:00Z)"""

        data = extract_data_from_influxdb(credentials, query)

        # Define column names
        column_names = ['date_time', 'value', 'measurement', 'field', 'sensor']

        # Create the DataFrame
        df = pd.DataFrame(data, columns=column_names)
        df_new = []
        fields = ['tmp', 'CO2', 'BLE', 'IR', 'WIFI', 'VOC', 'bandwidth', 'channel_index', 'channel_rssi', 'f_cnt', 'hum', 'rssi', 'spreading_factor', 'vis', 'device_id', 'snr', 'gateway']
        for time in df['date_time'].unique():
            df_time = df.loc[df['date_time'] == time]
            row = [time]
            for field in fields:
                row.append(df_time[df_time['field'] == field]['value'].values[0])
            df_new.append(row)

        df_new = pd.DataFrame(df_new, columns=['date_time'] + fields)
    
        if filename == "agg_hourly.parquet":
            df_new = dp.group_data(df_new, freq="h") 

    try:
        filepath = os.path.join("./data/aggregated_data/", filename)
        df_new = pd.read_parquet(filepath)
    except FileNotFoundError:
        logging.error(f"File {filepath} does not exist.")
        raise

    return df_new


@st.cache_resource
def load_model(model_name: str, device: torch.device):
    """
    Load the pre-trained model from the models directory.

    args:   model_name: str
            device: torch.device
    returns:    torch.nn.Module
    """
    filename = f"{model_name}.pth"
    filepath = os.path.join("./main/models/", filename)

    try:

        if model_name == "LSTM":
            model = foo.LSTM(input_size= 4, hidden_size= 100, num_layers= 1, output_size= 1, dropout= 0)

        elif model_name == "Transformer":
            model = foo.Decoder(input= 4, d_model= 64, max_len= 50, num_heads= 4, d_ff= 100, device= device)

        model.load_state_dict(torch.load(filepath, map_location= device))

    except FileNotFoundError:
        logging.error(f"File {filepath} does not exist.")
        raise FileNotFoundError(f"File {filepath} does not exist.")

    return model

def update_features_and_targets(i, test_features, test_targets, horizon_dict, device) -> tuple:
    """
    Updates the features and targets for the next prediction step.

    args:   i: int
            test_features: torch.tensor
            test_targets: torch.tensor
            horizon_dict: dict
            device: torch.device
    returns: tuple
    """
    if i > 0:
        test_features_sliced = test_features[:, 1:, :].to(device)

        for idx, test_feature in enumerate(test_features_sliced):
            previous_prediction = horizon_dict[i-1][idx]
            
            if previous_prediction.dim() == 0:
                previous_prediction = previous_prediction.unsqueeze(0)

            new_feature = torch.cat((previous_prediction, test_feature[-1][1:]), dim=0).unsqueeze(0)
            test_features[idx] = torch.cat((test_feature, new_feature), dim=0)
        
        test_targets = torch.cat((test_targets[i:].to(device), torch.zeros(i, test_targets.shape[1]).to(device)), dim=0)

    return test_features.to(device), test_targets.to(device)

@st.cache_data
def create_Prediction(filepath: str= "agg_hourly.parquet", model_name: str= "LSTM", scaling: bool= True, target: str= "tmp", features: list=["tmp", "hum", "VOC", "CO2"], device_ids: list= ["a017", "a014", "a101", "a102", "a103", "a106", "a107", "a108", "a111", "a112"], start_date: str= "", horizon_step: int= 1) -> list:
    """
    Create a prediction from a pre-trained model for a given time range.

    args:   filepath: str
            model: str
            scaling: bool
            target: str
            features: list
            device_ids: list
            start_date: str
            horizon_step: int
    returns: list
    """
    data = load_data(filename= filepath)
    df = deepcopy(data)

    start_date = pd.to_datetime(start_date)
    df = df[df["date_time"].between(start_date, start_date + pd.Timedelta(hours= 64))]
        
    if df.empty:
        logging.error("No data available for the selected date range.")
        output = [torch.tensor([]), torch.tensor([]), torch.tensor([])]    

    else:
        df = dp.build_lvl_df(df, device_ids= device_ids, output_cols= features, reset_ind= True)
        df["target"] = df[f"{target}"].shift(-1)

        if scaling:
            scaler = StandardScaler()
            df = scaler.fit_transform(df)

        X = dp.format_tensor(torch.tensor(df[:, :-1], dtype= torch.float32), 50)
        y = torch.tensor(df[:-1, -1], dtype= torch.float32).view(-1, 1)
        
        data_loader = DataLoader(TensorDataset(X, y), shuffle=False, batch_size=64) 
        test_features, test_targets = next(iter(data_loader))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_name, device)
        model.to(device)
        model.eval()

        horizon_dict = {}
        for i in range(horizon_step):
            with torch.no_grad():
                test_features, test_targets = update_features_and_targets(i, test_features, test_targets, horizon_dict, device)
                predictions = model(test_features)
                horizon_dict[i] = predictions

        test_loss = nn.MSELoss()(horizon_dict[horizon_step-1], test_targets.squeeze(-1)) if model_name == "Transformer" else nn.MSELoss()(horizon_dict[horizon_step-1], test_targets) 

        if scaling:
            feature_index = 0
            feature_scaler = StandardScaler()
            feature_scaler.mean_ = scaler.mean_[feature_index]
            feature_scaler.scale_ = scaler.scale_[feature_index]

            targets = feature_scaler.inverse_transform(test_targets.to("cpu").numpy().reshape(-1, 1))
            predictions = feature_scaler.inverse_transform(horizon_dict[horizon_step-1].to("cpu").numpy().reshape(-1, 1))
            
            test_loss = nn.MSELoss()(torch.from_numpy(predictions[:-horizon_step]), torch.from_numpy(targets[:-horizon_step]))
        
        output = [predictions, targets, test_loss]

    return output

@st.cache_data
def create_ensemble(model_predictions, targets) -> list:
    """
    Create an ensemble prediction from the LSTM and the Transformer model.

    args: model_predictions: list of numpy arrays or tensors
    return: tensor
    """
    predictions_np = [pred.numpy() if isinstance(pred, torch.Tensor) else pred for pred in model_predictions]
    predictions_np = [pred.reshape(-1, 1) for pred in predictions_np]

    predictions_tensor = torch.stack([torch.from_numpy(pred) for pred in predictions_np])

    ensemble = torch.mean(predictions_tensor, dim=0)
    ensemble_loss = nn.MSELoss()(ensemble, torch.from_numpy(targets))

    return [ensemble, ensemble_loss]

def plot_predictions(x, targets, predictions, loss):
    """
    Plot the predictions and the targets.

    args:   x: pd.DatetimeIndex
            targets: torch.tensor
            predictions: torch.tensor
            loss: torch.tensor
    return: st.plotly_chart, st.write
    """
    targets_size = targets.size if isinstance(targets, np.ndarray) else targets.numel()
    predictions_size = predictions.size if isinstance(predictions, np.ndarray) else predictions.numel()

    if targets_size == 0 or predictions_size == 0:
        logging.error("No data available for the selected date range.")
        with st.expander("ERROR: See explanation"):
            st.write(f"No data available for the selected date. To see what data is available, see 'prediction/Tmp gaps'.") 

    else:
        fig = go.Figure(
            data=[
                go.Scatter(x=x, y=targets.reshape(-1).tolist(), name="Targets", mode="lines"),
                go.Scatter(x=x, y=predictions.reshape(-1).tolist(), name="Predictions", mode="lines", line=dict(color="red"))
            ],
            layout=dict(title="Temperature in °C with predictions")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("loss: ", loss.item())

def prepare_data(input_device, df, key_prefix) -> pd.DataFrame:
    df_gaps = dp.build_lvl_df(df, a0 + a1, output_cols=OUTPUT_COLS, reset_ind=False).reset_index(drop=False) if input_device == "all" else df[(df["device_id"].astype(str) == input_device)]
    min_date = df_gaps["date_time"].min().date()
    max_date = df_gaps["date_time"].max().date()

    selected_range = st.slider("Select a range", key=f"slider_{key_prefix}", min_value=min_date, max_value=max_date, value=((min_date), (max_date)))
    selected_range = pd.to_datetime(selected_range[0]), pd.to_datetime(selected_range[1])
    df_filtered = df_gaps[(df_gaps["date_time"] >= selected_range[0]) & (df_gaps["date_time"] <= selected_range[1])]

    return df_filtered

# Sidebar
st.sidebar.header("Prediction Dashboard Building A")
input_use_influx_db_data = st.sidebar.checkbox(label= "Use InfluxDB data", value= False)


# Load data
a0 = ["a017", "a014"]
a1 = ["a101", "a102", "a103", "a106", "a107", "a108", "a111", "a112"]
OUTPUT_COLS = ["tmp", "hum", "snr", "CO2", "VOC", "vis", "IR", "WIFI", "BLE", "rssi", "channel_rssi", "channel_index", "spreading_factor", "bandwidth", "f_cnt"]
FILENAME = "agg_hourly.parquet"

data = load_data(FILENAME, use_influx_db= input_use_influx_db_data)
df = deepcopy(data)



# Page Content
tmp_tab1, tab_pred = st.tabs(["Tmp gaps", "Tmp pred"])


with tmp_tab1:
    input_device = st.selectbox(label= "Select Room", key= "roompicker_tab1", options= data["device_id"].unique().tolist() + ["all"], index= 2)
    input_trend = st.selectbox(label= "Select Trendline", key= "trendpicker_tab1", options= [False, True], index= 0)
    df_filtered = prepare_data(input_device, df, "tab1")
    st.markdown("### Temperature in °C from" + f" {df_filtered['date_time'].min().date()} to {df_filtered['date_time'].max().date()}")
    
    if input_trend:
        st.plotly_chart(dp.plt_fig(df_filtered, "tmp", trendline=True), use_container_width=True)
    
    else:
        st.plotly_chart(dp.plt_fig(df_filtered, "tmp", "markers"), use_container_width=True)

    st.dataframe(df_filtered)


with tab_pred:

    input_pred_model = st.selectbox(label= "Select Model", options= ["LSTM", "Transformer", "Ensemble"], index= 0)

    input_pred_start_date = st.date_input(label= "Select Start Date", value= date(2023,9,4), min_value= data["date_time"].min().date(), max_value= data["date_time"].max().date())
    input_pred_start_time = st.time_input(label= "Select Start Time", value= time(0, 0))
    input_pred_start_datetime = datetime.combine(input_pred_start_date, input_pred_start_time)
    
    input_pred_step_size = st.number_input(label= "Select Horizon Step", value= 2, min_value= 1, max_value= 4)

    if input_pred_model == "LSTM":
        pred_data = create_Prediction(filepath= FILENAME, model_name= "LSTM", scaling= True, start_date= input_pred_start_datetime, horizon_step= input_pred_step_size)
        inversed_predicitons = pred_data[0]
        inversed_targets = pred_data[1] 
        loss = pred_data[2]

        x = pd.date_range(start= input_pred_start_datetime, end= input_pred_start_datetime + pd.Timedelta(hours= 64), freq= "H")
        plot_predictions(x, inversed_targets, inversed_predicitons, loss)

    elif input_pred_model == "Transformer":    
        pred_data = create_Prediction(filepath= FILENAME, model_name= "Transformer", scaling= True, start_date= input_pred_start_datetime, horizon_step= input_pred_step_size)
        predictions = pred_data[0]
        targets = pred_data[1] 
        loss = pred_data[2]

        x = pd.date_range(start= input_pred_start_date, end= input_pred_start_datetime + pd.Timedelta(hours= 64), freq= "H")
        plot_predictions(x, targets, predictions, loss)

    elif input_pred_model == "Ensemble":    
        pred_data_lstm = create_Prediction(filepath= FILENAME, model_name= "LSTM", scaling= True, start_date= input_pred_start_datetime, horizon_step= input_pred_step_size)
        pred_data_trsnf = create_Prediction(filepath= FILENAME, model_name= "Transformer", scaling= True, start_date= input_pred_start_datetime, horizon_step= input_pred_step_size)
        predictions_lstm, targets_lstm, loss_lstm = pred_data_lstm[0], pred_data_lstm[1], pred_data_lstm[2]
        predictions_trsnf, targets_trsnf, loss_trnsf = pred_data_trsnf[0], pred_data_trsnf[1], pred_data_trsnf[2]
        ensemble = create_ensemble([predictions_lstm, predictions_trsnf], targets_lstm)
        ensemble_vec, ensemble_loss = ensemble[0], ensemble[1]

        x = pd.date_range(start= input_pred_start_datetime, end= input_pred_start_datetime + pd.Timedelta(hours= 64), freq= "H")
        plot_predictions(x, targets_lstm, ensemble_vec, ensemble_loss)


    
