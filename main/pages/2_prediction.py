import streamlit as st, pandas as pd, plotly.graph_objects as go, dataprep as dp, torch.nn as nn
import torch
import foo
import logging
import os
from datetime import datetime, date, time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
logging.basicConfig(level=logging.INFO)


# Page config
st.set_page_config(
    page_title="Sensor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Functions
@st.cache_data
def load_data(filename: str = "agg_hourly.parquet") -> pd.DataFrame:
    """
    Load data from the aggregated_data folder.

    args:   filename: str
    returns: pd.DataFrame
    """
    filepath = os.path.join("./data/aggregated_data/", filename)
    if not os.path.exists(filepath):
        logging.error(f"File {filepath} does not exist.")
        raise FileNotFoundError(f"File {filepath} does not exist.")
    df = pd.read_parquet(filepath)
    return df

@st.cache_resource
def load_model(model_name: str, device: torch.device):
    """
    Load the pre-trained model from the models directory.

    args:   model_name: str
            device: torch.device
    returns:    torch.nn.Module
    """
    print("device", device)
    filename = f"{model_name}.pth"
    filepath = os.path.join("./main/models/", filename)
    try:
        if model_name == "LSTM":
            model = foo.LSTM(input_size= 4, hidden_size= 100, num_layers= 1, output_size= 1, dropout= 0)
        elif model_name == "Transformer":
            model = foo.Decoder(input= 4, d_model= 64, max_len= 50, num_heads= 4, d_ff= 100, device= device)
        model.load_state_dict(torch.load(filepath, map_location= device))
        model.eval()
    except FileNotFoundError:
        logging.error(f"File {filepath} does not exist.")
        raise
    return model

@st.cache_data
def create_Prediction(filepath: str= "agg_hourly.parquet", model_name: str= "LSTM", scaling: bool= True, target: str= "tmp", features: list=["tmp", "hum", "VOC", "CO2"], device_ids: list= ["a017", "a014", "a101", "a102", "a103", "a106", "a107", "a108", "a111", "a112"], start_date: str= "", horizon_step: int= 2) -> list:
    """
    Create a prediction from a pre-trained model for a given time range.

    args:   filepath: str
            model: str
            scaling: bool
            target: str
            features: list
            start_date: str
            end_date: str
            device_ids: list
    returns: list
    """
    data = load_data(filename= filepath)
    df = deepcopy(data)

    start_date = pd.to_datetime(start_date)
    df = df[df["date_time"].between(start_date, start_date + pd.Timedelta(hours= 64))]
    if df.empty:
        logging.error("Keine Daten für den angegebenen Zeitraum gefunden.")
        raise ValueError("Keine Daten für den angegebenen Zeitraum gefunden.")
        

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
            if i > 0:
                test_features_new = test_features[:,1:,:].to(device)
            test_features_new2 = test_features
            if i > 0:
                for idx,test_feature in enumerate(test_features_new):
                    if horizon_dict[i-1][idx].dim() == 0:
                        test_features_new2[idx] = torch.cat((test_feature,torch.cat((horizon_dict[i-1][idx].unsqueeze(0), test_feature[-1][1:]), dim=0).unsqueeze(0)), dim=0)
                    else:
                        test_features_new2[idx] = torch.cat((test_feature,torch.cat((horizon_dict[i-1][idx], test_feature[-1][1:]), dim=0).unsqueeze(0)), dim=0)
            test_targets = test_targets[i:].to(device)
            if i > 0:
                zeros = torch.zeros(i, test_targets.shape[1]).to(device)
                test_targets = torch.cat((test_targets, zeros), dim=0)
            test_features = test_features_new2
            test_features_new2 = test_features_new2.to(device)
            predictions = model(test_features_new2)
            horizon_dict[i] = predictions

    test_loss = nn.MSELoss()(horizon_dict[horizon_step-1], test_targets.squeeze(-1)) if model_name == "Transformer" else nn.MSELoss()(horizon_dict[horizon_step-1], test_targets) 
    print("test_loss", test_loss)
    if scaling:
        feature_index = 0
        feature_scaler = StandardScaler()
        feature_scaler.mean_ = scaler.mean_[feature_index]
        feature_scaler.scale_ = scaler.scale_[feature_index]

        targets = feature_scaler.inverse_transform(test_targets.to("cpu").numpy().reshape(-1, 1))
        predictions = feature_scaler.inverse_transform(horizon_dict[horizon_step-1].to("cpu").numpy().reshape(-1, 1))
        test_loss = nn.MSELoss()(torch.from_numpy(predictions), torch.from_numpy(targets))
        print("test_loss2", test_loss)
    output = [predictions, targets, test_loss]
    #print(predictions)

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
    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=targets.reshape(-1).tolist(), name="Targets", mode="lines"),
            go.Scatter(x=x, y=predictions.reshape(-1).tolist(), name="Predictions", mode="lines", line=dict(color="red"))
        ],
        layout=dict(title="Temperature in °C with predictions")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("loss: ", loss.item())

def prepare_data(input_device, df, key_prefix):
    df_gaps = dp.build_lvl_df(df, a0 + a1, output_cols=OUTPUT_COLS, reset_ind=False).reset_index(drop=False) if input_device == "all" else df[(df["device_id"].astype(str) == input_device)]
    min_date = df_gaps["date_time"].min().date()
    max_date = df_gaps["date_time"].max().date()

    selected_range = st.slider("Select a range", key=f"slider_{key_prefix}", min_value=min_date, max_value=max_date, value=((min_date), (max_date)))
    selected_range = pd.to_datetime(selected_range[0]), pd.to_datetime(selected_range[1])
    df_filtered = df_gaps[(df_gaps["date_time"] >= selected_range[0]) & (df_gaps["date_time"] <= selected_range[1])]

    return df_filtered

# Load data
a0 = ["a017", "a014"]
a1 = ["a101", "a102", "a103", "a106", "a107", "a108", "a111", "a112"]
FILENAME = "agg_hourly.parquet"
OUTPUT_COLS = ["tmp", "hum", "snr", "CO2", "VOC", "vis", "IR", "WIFI", "BLE", "rssi", "channel_rssi", "channel_index", "spreading_factor", "bandwidth", "f_cnt"]

data = load_data(FILENAME)
df = deepcopy(data)

# Sidebar
st.sidebar.header("Prediction Dashboard Building A")

# Page Content
tmp_tab1, tab_trend, tab_pred = st.tabs(["Tmp gaps", "Tmp trend", "Tmp pred"])


with tmp_tab1:
    input_device = st.selectbox(label= "Select Room", key= "roompicker_tab1", options= data["device_id"].unique().tolist() + ["all"], index= 2)
    df_filtered = prepare_data(input_device, df, "tab1")
    st.markdown("### Temperature in °C seit Aufzeichnungsbeginn")
    st.plotly_chart(dp.plt_fig(df_filtered, "tmp", "markers"), use_container_width=True)
    st.dataframe(df_filtered)


with tab_trend:
    input_device = st.selectbox(label= "Select Room", key= "roompicker_tab2", options= data["device_id"].unique().tolist() + ["all"], index= 2)
    df_filtered = prepare_data(input_device, df, "tab2")
    st.markdown("### Temperature in °C mit Trendline")
    st.plotly_chart(dp.plt_fig(df_filtered, "tmp", trendline=True), use_container_width=True)
    st.dataframe(df_filtered)


with tab_pred:

    input_pred_model = st.selectbox(label= "Select Model", options= ["LSTM", "Transformer", "Ensemble"], index= 0)
    #input_pred_start_date = st.date_input(label= "Select Start Date", value= date(2023,9,4), min_value= data["date_time"].min(), max_value= data["date_time"].max())
    #input_pred_start_date = st.date_input(label= "Select End Date", value= date(2023,10,1), min_value= data["date_time"].min(), max_value= data["date_time"].max())
    input_pred_start_date = st.date_input(label= "Select Start Date", value= date(2023,9,4), min_value= data["date_time"].min().date(), max_value= data["date_time"].max().date())
    input_pred_start_time = st.time_input(label= "Select Start Time", value= time(0, 0))
    input_pred_start_datetime = datetime.combine(input_pred_start_date, input_pred_start_time)
    input_pred_step_size = st.number_input(label= "Select Horizon Step", value= 2, min_value= 1, max_value= 4)

    if input_pred_model == "LSTM":
        pred_data = create_Prediction(filepath= FILENAME, model_name= "LSTM", scaling= True, start_date= input_pred_start_datetime, horizon_step= input_pred_step_size)
        inversed_predicitons = pred_data[0]
        inversed_targets = pred_data[1] 
        loss = pred_data[2]

        #x = pd.date_range(start= input_pred_start_date, end= pred_end_date)
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


    
