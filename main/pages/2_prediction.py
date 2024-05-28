import streamlit as st, pandas as pd, numpy as np
import plotly.graph_objects as go
from datetime import date
import dataprep as dp
from foo import evaluate_model



# Load data
a0 = ["hka-aqm-a017", "hka-aqm-a014"]
a1 = ["hka-aqm-a101", "hka-aqm-a102", "hka-aqm-a103", "hka-aqm-a106", "hka-aqm-a107", "hka-aqm-a108", "hka-aqm-a111", "hka-aqm-a112"]
filename = "/Users/florian/Documents/github/study/IoT/IoT/main/output.csv"

@st.cache
def load_data(filename: str="output.csv", device_ids: list=[]):
    """
    args:   filename: csv file to read data from
            device_ids: list of device ids to filter and process data

    returns: processed dataframe
    """
    data = pd.read_csv(filename)
    df_hour = dp.group_data(data, "h")
    df_hour = df_hour[df_hour["device_id"].isin(device_ids)]  
    df_hour["device_id"] = df_hour["device_id"].str.replace("hka-aqm-", "")
    return df_hour

@st.cache
def prep_eval_data(filename: str="aggregated_hourly.csv", device_ids: list=[], columns: list= ["tmp", "hum", "CO2", "VOC"]):
    """
    args:   filename: csv file to read data from
            device_ids: list of device ids to filter data
            columns: columns to select from the data

    returns: dictionary of dataframes
    """
    df = pd.read_csv(filename)
    df.date_time = pd.to_datetime(df.date_time)
    prep_dev_df = {device_id: df[df["device_id"] == device_id][columns+["date_time"]] for device_id in device_ids}
    return prep_dev_df

df_hour = load_data(a0 + a1)
prep_dfs_a0 = prep_eval_data(device_ids= a0)
prep_dfs_a1 = prep_eval_data(device_ids= a1)



# Sidebar
st.sidebar.header("Sensor Dashboard Building A")
input_device = st.sidebar.selectbox(label= "Select Room", options= df_hour["device_id"].unique().tolist(), index= 2)
input_date = st.sidebar.date_input(label= "Select Date", value= date(2022,10,10), min_value= df_hour["date_time"].min(), max_value= df_hour["date_time"].max())

# Filter data
df_device_dt = df_hour[(df_hour["device_id"].astype(str) == input_device) & (df_hour["date_time"].astype(str).str.slice(0, 10).str.contains(str(input_date)))]
df_gaps = df_hour[df_hour["device_id"].astype(str) == input_device]


# Gaps in the data and predictions
st.markdown("## Gaps in the Data")
tmp_tab1, hum_tab2, co2_tab3 = st.tabs(["Tmp gaps", "Tmp trend", "Tmp pred"])

with tmp_tab1:
    st.markdown("### Temperature in °C seit Aufzeichnungsbeginn")
    st.plotly_chart(dp.plt_fig(df_gaps, "tmp", "markers"), use_container_width=True)
    
with hum_tab2:
    st.markdown("### Temperature in °C mit Trendline")
    st.plotly_chart(dp.plt_fig(df_gaps, "tmp", trendline=True), use_container_width=True)

with co2_tab3:
    st.markdown("### CO2 in ppm")
    


# Detailed data view
st.markdown("## Detailed Data View") 
st.dataframe(df_device_dt)
