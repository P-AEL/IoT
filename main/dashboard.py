import streamlit as st, pandas as pd, pyvista as pv, dataprep as dp
from stpyvista import stpyvista
from datetime import date
#from pyvista_test import plot_cube
import os
import logging
logging.basicConfig(level=logging.INFO)


# Page config
st.set_page_config(
    page_title="Sensor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data 
def load_data(filepath: str= "output.parquet"):
    if not os.path.exists(filepath):
        logging.error(f"File {filepath} does not exist.")
        raise FileNotFoundError(f"File {filepath} does not exist.")
    df = pd.read_parquet(filepath)
    return df

# Load data
filename = "/Users/florian/Documents/github/study/IoT/IoT/main/agg_hourly.parquet"
df = load_data(filename)

# Sidebar
st.sidebar.header("Sensor Dashboard Building A")
input_device = st.sidebar.selectbox(label= "Select Room", options= df["device_id"].unique().tolist(), index= 2)
input_date = st.sidebar.date_input(label= "Select Date", value= date(2022,10,10), min_value= df["date_time"].min(), max_value= df["date_time"].max())

# Filter data
df_device_dt = df[(df["device_id"].astype(str) == input_device) & (df["date_time"].astype(str).str.slice(0, 10).str.contains(str(input_date)))]


# Columns
c1, c2 = st.columns(2)

with c1:

    st.markdown("## Pauls Tolle Graphik")
    # plotter = plot_cube()
    # plotter.view_isometric()
    # plotter.add_scalar_bar()
    # plotter.background_color = "black"
    # stpyvista(plotter)

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
        with st.expander("See explanation"):
            st.write(f"No data available for the selected room {input_device} and room {input_date}. To see what data is available, look 'Gaps in the data'.")         

    
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

