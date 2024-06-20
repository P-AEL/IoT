import streamlit as st, pandas as pd, pyvista as pv, dataprep as dp
from stpyvista import stpyvista
from datetime import date
from pyvista_test import plot_cube
from copy import deepcopy


# Page config
st.set_page_config(
    page_title="Sensor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
FILENAME = "agg_hourly.parquet"
data = dp.load_data(FILENAME)
df = deepcopy(data)

# Sidebar
st.sidebar.header("Sensor Dashboard Building A")
input_device = st.sidebar.selectbox(label= "Select Room", options= df["device_id"].unique().tolist() + ["all"], index= 2)
input_date = st.sidebar.date_input(label= "Select Date", value= date(2022,10,10), min_value= df["date_time"].min(), max_value= df["date_time"].max())
input_use_influx_db_data = st.sidebar.checkbox(label= "Use InfluxDB data", value= False)


# Filter data
a0 = ["a017", "a014"]
a1 = ["a101", "a102", "a103", "a106", "a107", "a108", "a111", "a112"]
FILENAME = "agg_hourly.parquet"
OUTPUT_COLS = ["tmp", "hum", "snr", "CO2", "VOC", "vis", "IR", "WIFI", "BLE", "rssi", "channel_rssi", "channel_index", "spreading_factor", "bandwidth", "f_cnt"]


if input_device == "all":
    df_temp = dp.build_lvl_df(df, a0 + a1, output_cols=OUTPUT_COLS, reset_ind=False).reset_index(drop=False)
    mask = df_temp["date_time"].astype(str).str.slice(0, 10).str.contains(str(input_date))
    df_device_dt = df_temp[mask]
else:
    mask = (df["device_id"].astype(str) == input_device) & (df["date_time"].astype(str).str.slice(0, 10).str.contains(str(input_date)))
    df_device_dt = df[mask]


# Page content
c1, c2 = st.columns(2)

with c1:

    st.markdown("## Building A")
    # plotter = plot_cube(input_device)
    # plotter.view_isometric()
    # plotter.add_scalar_bar()
    # plotter.background_color = "black"
    # # Set the camera position
    # plotter.camera_position = ((2.5, 2.5, 2.5), (0.3, 0.5, -0.5), (0, 0, 1))

    # stpyvista(plotter)
    # if input_device == "a014":
    #     st.write("We didnt find room 014, it possible doesn't exist anymore.")

with c2:

    kpi_columns = st.columns(3)
    kpi_dict = {
        "Temperature": {"df_col": "tmp", "unit": "°C"},
        "Humidity": {"df_col": "hum", "unit": "%"},
        "CO2": {"df_col": "CO2", "unit": "ppm"},
        "VOC": {"df_col": "VOC", "unit": "ppb"},
        "Wifi": {"df_col": "WIFI", "unit": "num"},
        "rssi": {"df_col": "rssi", "unit": "dbm"}
    }

    kpi_items = list(kpi_dict.items())
    for i, column in enumerate(kpi_columns):
        with column:
            for j in range(2):
                name, info = kpi_items[i*2 + j]
                with st.container(border=True):
                    value = f"{df_device_dt[info['df_col']].mean().round(2)} {info['unit']}" if not df_device_dt.empty else "n/a"
                    st.metric(name, value)

    if df_device_dt.empty:
        with st.expander("ERROR: See explanation"):
            st.write(f"No data available for the selected room {input_device} and date {input_date}. To see what data is available, see 'prediction/Tmp gaps'.")         

    
    tabs = st.tabs(["Temperature", "Humidity", "CO2", "VOC", "Wifi", "rssi"])
    tab_info = {
        "Temperature": {"key": "tmp", "unit": "°C"},
        "Humidity": {"key": "hum", "unit": "%"},
        "CO2": {"key": "CO2", "unit": "ppm"},
        "VOC": {"key": "VOC", "unit": "ppb"},
        "Wifi": {"key": "WIFI", "unit": "num"},
        "rssi": {"key": "rssi", "unit": "dbm"}
    }

    for tab, (name, info) in zip(tabs, tab_info.items()):
        with tab:
            st.markdown(f"### {name} in {info['unit']}")
            st.plotly_chart(dp.plt_fig(df_device_dt, info['key']))

