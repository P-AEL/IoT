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

    try:
        kpi_tmp, kpi_hum, kpi_co2 = st.columns(3)

        with kpi_tmp:
            container = st.container(border=True)
            with container:
                st.metric("Temperature", f"{df_device_dt['tmp'].mean().round(2)} °C")

        with kpi_hum:
            container = st.container(border=True)
            with container:
                st.metric("Humidity", f"{df_device_dt['hum'].mean().round(2)} %")

        with kpi_co2:
            container = st.container(border=True)
            with container:
                st.metric("CO2", f"{df_device_dt['CO2'].mean().round(2)} ppm")   

    except AttributeError:
        st.markdown("## No data available")
        with st.expander("See explanation"):
            st.write(f"No data available for the selected room {input_device} and room {input_date}. To see what data is available, look 'Gaps in the data'.")              

    tab_tmp, tab_hum, tab_co2, tab_voc, tab_wifi = st.tabs(["Temperature", "Humidity", "CO2", "VOC", "WIFI"])

    with tab_tmp:
        st.markdown("### Temperature in °C")
        st.plotly_chart(dp.plt_fig(df_device_dt, "tmp"))
    
    with tab_hum:
        st.markdown("### Humidity in %")
        st.plotly_chart(dp.plt_fig(df_device_dt, "hum"))

    with tab_co2:
        st.markdown("### CO2 in ppm")
        st.plotly_chart(dp.plt_fig(df_device_dt, "CO2"))

    with tab_voc:
        st.markdown("### VOC in ppb")
        st.plotly_chart(dp.plt_fig(df_device_dt, "VOC"))

    with tab_wifi:
        st.markdown("### Num of WIFI Devices detected")
        st.plotly_chart(dp.plt_fig(df_device_dt, "WIFI"))



# To do:
# raum102 wirft fehler weil keine daten vorhanden -> kpi wirft round fehler
# mehr tabs hinzufügen
# dashboard anpassen