import streamlit as st, pandas as pd, numpy as np
import plotly.graph_objects as go
from datetime import date
import dataprep as dp



# Load data
data = pd.read_csv("/Users/florian/Documents/github/study/IoT/IoT/main/output.csv")
df_hour = dp.group_data(data, "h")

a0 = ["hka-aqm-a017", "hka-aqm-a014"]
a1 = ["hka-aqm-a101", "hka-aqm-a102", "hka-aqm-a103", "hka-aqm-a106", "hka-aqm-a107", "hka-aqm-a108", "hka-aqm-a111", "hka-aqm-a112"]
df_hour = df_hour[df_hour["device_id"].isin(a0 + a1)]  
df_hour["device_id"] = df_hour["device_id"].str.replace("hka-aqm-", "")

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
    st.plotly_chart(dp.plt_fig(df_gaps, "CO2", "markers+lines"), use_container_width=True)


# Detailed data view
st.markdown("## Detailed Data View") 
st.dataframe(df_device_dt)
