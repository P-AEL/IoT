import streamlit as st, pandas as pd, numpy as np, pyvista as pv
import plotly.graph_objects as go
from stpyvista import stpyvista
from datetime import date
import dataprep as dp
from scipy import stats
#from pyvista_test import plot_cube

# Page config
st.set_page_config(
    page_title="Sensor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
    )
st.title("Sensor Dashboard Building A")
# menu_items={
# ...         'Get Help': 'https://www.extremelycoolapp.com/help',
# ...         'Report a bug': "https://www.extremelycoolapp.com/bug",
# ...         'About': "# This is a header. This is an *extremely* cool app!"
# ...     }

# Load data
data = pd.read_csv("/Users/florian/Documents/github/study/IoT/IoT/main/output.csv")
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

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        container = st.container(border=True)
        with container:
            st.metric("Temperature", f"{df_device_dt['tmp'].mean().round(2)} °C")
    
    with kpi2:
        container = st.container(border=True)
        with container:
            st.metric("Humidity", f"{df_device_dt['hum'].mean().round(2)} %")

    with kpi3:
        container = st.container(border=True)
        with container:
            st.metric("CO2", f"{df_device_dt['CO2'].mean().round(2)} ppm")                    

    tab1, tab2, tab3 = st.tabs(["Temperature", "Humidity", "CO2"])

    with tab1:
        st.markdown("### Temperature in °C")
        st.plotly_chart(dp.plt_fig(df_device_dt, "tmp"))
       
    with tab2:
        st.markdown("### Humidity in %")
        st.plotly_chart(dp.plt_fig(df_device_dt, "hum", trendline=True))

    with tab3:
        st.markdown("### CO2 in ppm")
        st.plotly_chart(dp.plt_fig(df_device_dt, "CO2"))


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



# To do:
# raum102 wirft fehler weil keine daten vorhanden -> kpi wirft round fehler
# mehr tabs hinzufügen
# dashboard anpassen