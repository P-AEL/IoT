import streamlit as st, pandas as pd, numpy as np, pyvista as pv
from stpyvista import stpyvista
from datetime import date
import dataprep as dp
#from pyvista_test import plot_cube

# Page config
st.set_page_config(
    page_title="Sensor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data 
def load_data(filepath: str= "output.parquet"):
    df = pd.read_parquet(filepath)
    return df

# st.chache_resource for ml models and database resources

# Load data
data = load_data("/Users/florian/Documents/github/study/IoT/IoT/main/output.parquet")
df_hour = dp.group_data(data, "h")

# Sidebar
st.sidebar.header("Sensor Dashboard Building A")
input_device = st.sidebar.selectbox(label= "Select Room", options= df_hour["device_id"].unique().tolist(), index= 2)
input_date = st.sidebar.date_input(label= "Select Date", value= date(2022,10,10), min_value= df_hour["date_time"].min(), max_value= df_hour["date_time"].max())

# Filter data
df_device_dt = df_hour[(df_hour["device_id"].astype(str) == input_device) & (df_hour["date_time"].astype(str).str.slice(0, 10).str.contains(str(input_date)))]
df_gaps = df_hour[df_hour["device_id"].astype(str) == input_device]


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
    }

    for column, (name, info) in zip(kpi_columns, kpi_dict.items()):
        with column, st.container(border=True):
            value = f"{df_device_dt[info['df_col']].mean().round(2)} {info['unit']}" if not df_device_dt.empty else "n/a"
            st.metric(name, value)

    if df_device_dt.empty:
        with st.expander("See explanation"):
            st.write(f"No data available for the selected room {input_device} and room {input_date}. To see what data is available, look 'Gaps in the data'.")         

    
    tabs = st.tabs(["Temperature", "Humidity", "CO2", "VOC"])
    tab_info = {
        "Temperature": {"key": "tmp", "unit": "°C"},
        "Humidity": {"key": "hum", "unit": "%"},
        "CO2": {"key": "CO2", "unit": "ppm"},
        "VOC": {"key": "VOC", "unit": "ppb"},
    }

    for tab, (name, info) in zip(tabs, tab_info.items()):
        with tab:
            st.markdown(f"### {name} in {info['unit']}")
            st.plotly_chart(dp.plt_fig(df_device_dt, info['key']))




# To do:
# app nightly
    # 3 Kpi metrics hinzufügen wifi voc 
    # wifi als tab hinzufügen 
