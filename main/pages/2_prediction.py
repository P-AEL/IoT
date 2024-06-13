import streamlit as st, pandas as pd, numpy as np
import plotly.graph_objects as go
from datetime import date
import dataprep as dp
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import torch.nn as nn 
import foo


# Page config
st.set_page_config(
    page_title="Sensor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
a0 = ["a017", "a014"]
a1 = ["a101", "a102", "a103", "a106", "a107", "a108", "a111", "a112"]
filename_data = "/Users/florian/Documents/github/study/IoT/IoT/main/output.parquet"


@st.cache_data 
def load_data(filepath: str= "output.parquet") -> pd.DataFrame:
    """
    args: filepath: str
    return: pd.DataFrame
    """
    df = pd.read_parquet(filepath)
    return df


@st.cache_data
def create_Prediction(filename : str= "", rooms : list= [], agg : str= "h", start_date : str= "", end_date : str= "", features : list= ["tmp", "hum", "CO2", "VOC"], target : str= "tmp", train_size : float= 0.8, batch_size : int= 120, pred_model : str = "LSTM") -> list:
    """
    args:  filename: str, rooms: list, agg: str, start_date: str, end_date: str, features: list, target: str, train_size: float, batch_size: int, pred_model: str
    return: list
    """ 
    df = load_data(filename)
    df_rooms = df[df["device_id"].isin(rooms)]
    df = dp.group_data(df_rooms, agg)

    df_cutoff = df.copy
    df_cutoff = dp.cutoff_data(df, start_date, end_date)

    df_mean = dp.build_lvl_df(df_cutoff, rooms, features, reset_ind= True)


    if pred_model == "LSTM":

        scaler = StandardScaler()
        df_mean_scaled = deepcopy(df_mean)
        df_mean_scaled["target"] = df_mean_scaled[f"{target}"].shift(-1)
        df_mean_scaled = scaler.fit_transform(df_mean_scaled)

        X = df_mean_scaled[:, :-1]
        y = df_mean_scaled[:, -1]

        X_train, X_test, y_train, y_test = dp.train_test_split(X, y, train_size= train_size)

        X_train_new, X_test_new = dp.format_tensor(X_train, 100), dp.format_tensor(X_test, 100)
        X_train = X_train_new
        X_test = X_test_new
        y_train = y_train[:-1]
        y_test = y_test[:-1]

        #train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=False, batch_size=batch_size) 
        test_loader = DataLoader(TensorDataset(X_test, y_test), shuffle=False, batch_size=batch_size)

        model = foo.LSTM_1(input_size=X_train.shape[2], hidden_size=100, num_layers=1, output_size=1, dropout=0, activation='relu')
        model.load_state_dict(torch.load("/Users/florian/Documents/github/study/IoT/IoT/main/lstm.pth"))

        model.eval()  # Set the model to evaluation mode

        test_features, test_targets = next(iter(test_loader))  # Get a batch of train data
        test_targets = test_targets.unsqueeze(1)  # Expand target to match the output shape

        with torch.no_grad():  # Disable gradient computation
            predictions = model(test_features)  # Make predictions

        # Calculate the mean squared error of the predictions
        test_loss = nn.MSELoss()(predictions, test_targets)

        feature_index = 0
        # Erstellen Sie einen neuen `StandardScaler` für das entsprechende Feature
        feature_scaler = StandardScaler()
        feature_scaler.mean_ = scaler.mean_[feature_index]
        feature_scaler.scale_ = scaler.scale_[feature_index]

        # Verwenden Sie den `feature_scaler` um die Vorhersagen zurück zu transformieren
        inversed_predictions = feature_scaler.inverse_transform(predictions)
        # Tun Sie dasselbe für die Ziele
        inversed_targets = feature_scaler.inverse_transform(test_targets)

        output = [inversed_predictions, inversed_targets, test_loss]

    elif pred_model == "Transformer":

        X = df_mean.to_numpy()
        y = df_mean["tmp"].shift(-1).to_numpy()
        X_train, X_test, y_train, y_test = dp.train_test_split(X, y,train_size=0.8)
        X_train_new, X_test_new = dp.format_tensor(X_train,window_size=100), dp.format_tensor(X_test,window_size=100)
        y_train = y_train[:-1]
        y_test = y_test[:-1]

        X_train = X_train_new
        X_test = X_test_new
        
        #train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=False, batch_size=batch_size) 
        test_loader = DataLoader(TensorDataset(X_test, y_test), shuffle=False, batch_size=batch_size)
        
        model = foo.Decoder(4, 128, 100, 4, 256, device="cpu")
        model.to("cpu")
        model.load_state_dict(torch.load("/Users/florian/Documents/github/study/IoT/IoT/main/Decoder_besser.pth", map_location="cpu"))

        model.eval()

        test_features, test_targets = next(iter(test_loader))  # Get a batch of train data
        test_targets = test_targets.unsqueeze(1)  # Expand target to match the output shape

        with torch.no_grad():  # Disable gradient computation
            predictions = model(test_features)  # Make predictions

        # Calculate the mean squared error of the predictions
        test_loss = nn.MSELoss()(predictions, test_targets)

        output = [predictions, test_targets, test_loss]
   
    return output


def create_ensemble(model_predictions, targets):
    """
    args: model_predictions: list of numpy arrays or tensors
    return: tensor
    """
    # Konvertiert die Liste der Vorhersagen in Numpy-Arrays und stellt sicher, dass sie die richtige Form haben
    predictions_np = [pred.numpy() if isinstance(pred, torch.Tensor) else pred for pred in model_predictions]
    predictions_np = [pred.reshape(-1, 1) for pred in predictions_np]

    # Konvertiert die Liste der Vorhersagen zurück in Tensoren
    predictions_tensor = torch.stack([torch.from_numpy(pred) for pred in predictions_np])

    # Berechnet den Durchschnitt entlang der 0. Achse (Modellebene)
    ensemble = torch.mean(predictions_tensor, dim=0)
    ensemble_loss = nn.MSELoss()(ensemble, torch.from_numpy(targets))

    return [ensemble, ensemble_loss]



data = load_data(filename_data)
df_hour = dp.group_data(data, "h")

# Sidebar
st.sidebar.header("Sensor Dashboard Building A")
input_device = st.sidebar.selectbox(label= "Select Room", options= df_hour["device_id"].unique().tolist() + ["all"], index= 2)


# Filter data
df_gaps = dp.build_lvl_df(df_hour, a0 + a1, output_cols= ["tmp", "hum", "snr", "CO2", "VOC", "vis", "IR", "WIFI", "BLE", "rssi", "channel_rssi", "channel_index", "spreading_factor", "bandwidth", "f_cnt"], reset_ind= False).reset_index(drop= False) if input_device == "all" else df_hour[(df_hour["device_id"].astype(str) == input_device)]

# Konvertieren Sie pd.Timestamp in datetime.date für den Slider
min_date = df_gaps["date_time"].min().date()
max_date = df_gaps["date_time"].max().date()

tmp_tab1, tab_trend, tab_pred = st.tabs(["Tmp gaps", "Tmp trend", "Tmp pred"])

with tmp_tab1:

    selected_range = st.slider("Select a range", key= "slider1", min_value=min_date, max_value=max_date, value = ((min_date), (max_date)))

    # Konvertieren Sie datetime.date zurück in pd.Timestamp für die Filterung
    selected_range = pd.to_datetime(selected_range[0]), pd.to_datetime(selected_range[1])

    df_filtered = df_gaps[(df_gaps["date_time"] >= selected_range[0]) & (df_gaps["date_time"] <= selected_range[1])]

    st.markdown("### Temperature in °C seit Aufzeichnungsbeginn")
    st.plotly_chart(dp.plt_fig(df_filtered, "tmp", "markers"), use_container_width=True)
    st.dataframe(df_gaps)

    
with tab_trend:

    selected_range1 = st.slider("Select a range", key= "slider2", min_value=min_date, max_value=max_date, value = ((min_date), (max_date)))

    # Konvertieren Sie datetime.date zurück in pd.Timestamp für die Filterung
    selected_range1 = pd.to_datetime(selected_range1[0]), pd.to_datetime(selected_range1[1])

    df_filtered = df_gaps[(df_gaps["date_time"] >= selected_range1[0]) & (df_gaps["date_time"] <= selected_range1[1])]

    st.markdown("### Temperature in °C mit Trendline")
    st.plotly_chart(dp.plt_fig(df_filtered, "tmp", trendline=True), use_container_width=True)
    st.dataframe(df_gaps)


with tab_pred:

    input_pred_model = st.selectbox(label= "Select Model", options= ["LSTM", "Transformer", "Ensemble"], index= 0)
    pred_start_date = st.date_input(label= "Select Start Date", value= date(2023,9,4), min_value= df_hour["date_time"].min(), max_value= df_hour["date_time"].max())
    pred_end_date = st.date_input(label= "Select End Date", value= date(2023,10,1), min_value= df_hour["date_time"].min(), max_value= df_hour["date_time"].max())

    if input_pred_model == "LSTM":
        pred_data = create_Prediction(filename=filename_data, rooms= a1, agg= "h", start_date= str(pred_start_date), end_date= str(pred_end_date), features= ["tmp", "hum", "CO2", "VOC"], target= "tmp", train_size= 0.8, batch_size= 120, pred_model= "LSTM")
        inversed_predicitons = pred_data[0]
        inversed_targets = pred_data[1] 
        loss = pred_data[2]
        x = pd.date_range(start= pred_start_date, end= pred_end_date)
        x = x[(x >= selected_range[0]) & (x <= selected_range[1])]
        fig = go.Figure(
                data=[
                    go.Scatter(x= x,y=inversed_targets.reshape(-1).tolist(), name="Targets", mode="lines"),#, line=dict(color="blue")),
                    go.Scatter(x= x, y=inversed_predicitons.reshape(-1).tolist(), name="Predictions", mode="lines", line=dict(color="red"))
                ],
                layout=dict(title="Temperature in °C with predictions")
            )
        st.plotly_chart(fig,
            use_container_width=True
        )
        st.write("loss: ", loss.item())


    elif input_pred_model == "Transformer":    
        pred_data = create_Prediction(filename=filename_data, rooms= a1, agg= "h", start_date= str(pred_start_date), end_date= str(pred_end_date), features= ["tmp", "hum", "CO2", "VOC"], target= "tmp", train_size= 0.8, batch_size= 120, pred_model= "Transformer")
        predictions = pred_data[0]
        targets = pred_data[1] 
        loss = pred_data[2]
        x = pd.date_range(start= pred_start_date, end= pred_end_date)
        x = x[(x >= selected_range[0]) & (x <= selected_range[1])]
        fig = go.Figure(
                data=[
                    go.Scatter(x= x,y=targets.reshape(-1).tolist(), name="Targets", mode="lines"),#, line=dict(color="blue")),
                    go.Scatter(x= x, y=predictions.reshape(-1).tolist(), name="Predictions", mode="lines", line=dict(color="red"))
                ],
                layout=dict(title="Temperature in °C with predictions")
            )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
        st.write("loss: ", loss.item())


    elif input_pred_model == "Ensemble":    
        
        pred_data_lstm = create_Prediction(filename=filename_data, rooms= a1, agg= "h", start_date= str(pred_start_date), end_date= str(pred_end_date), features= ["tmp", "hum", "CO2", "VOC"], target= "tmp", train_size= 0.8, batch_size= 120, pred_model= "LSTM")
        pred_data_trsnf = create_Prediction(filename=filename_data, rooms= a1, agg= "h", start_date= str(pred_start_date), end_date= str(pred_end_date), features= ["tmp", "hum", "CO2", "VOC"], target= "tmp", train_size= 0.8, batch_size= 120, pred_model= "Transformer")
        predictions_lstm, targets_lstm, loss_lstm = pred_data_lstm[0], pred_data_lstm[1], pred_data_lstm[2]
        predictions_trsnf, targets_trsnf, loss_trnsf = pred_data_trsnf[0], pred_data_trsnf[1], pred_data_trsnf[2]


        ensemble = create_ensemble([predictions_lstm, predictions_trsnf], targets_lstm)
        ensemble_vec, ensemble_loss = ensemble[0], ensemble[1]

        x = pd.date_range(start= pred_start_date, end= pred_end_date)
        x = x[(x >= selected_range[0]) & (x <= selected_range[1])]
        fig = go.Figure(
                data=[
                    go.Scatter(x= x,y=targets_lstm.reshape(-1).tolist(), name="Targets", mode="lines"),#, line=dict(color="blue")),
                    go.Scatter(x= x, y=ensemble_vec.reshape(-1).tolist(), name="Predictions", mode="lines", line=dict(color="red"))
                ],
                layout=dict(title="Temperature in °C with predictions")
            )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
        st.write("loss: ", ensemble_loss.item())


    
