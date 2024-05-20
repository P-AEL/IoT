import streamlit as st, pandas as pd, numpy as np, pyvista as pv
import plotly.graph_objects as go
from group_data import group_data
from stpyvista import stpyvista
from datetime import date
import dataprep as dp
#from pyvista_test import plot_cube

# Set page config
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
data = pd.read_csv("output.csv")
df_hour = dp.group_data(data, "h")

# Sidebar
st.sidebar.header("Sensor Dashboard Building A")
input_device = st.sidebar.selectbox(label= "Select Room", options= df_hour["device_id"].unique().tolist(), index= 1)
input_date = st.sidebar.date_input(label= "Select Date", value= date(2022,10,10), min_value= df_hour["date_time"].min(), max_value= df_hour["date_time"].max())

# Filter data
filtered_data = df_hour[(df_hour["device_id"].astype(str) == input_device) & (df_hour["date_time"].astype(str).str.slice(0, 10).str.contains(str(input_date)))]

# Create plot
def plt_fig(data, y, mode="lines+markers"):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["date_time"],
                y=data[y],
                name=y,
                mode=mode
            )
        )
        fig.update_layout(xaxis_title= "Time", yaxis_title= y)
        return fig

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
        container = st.container()#border=True)
        with container:
            st.metric("Temperature", f"{filtered_data['tmp'].mean().round(2)} °C")
    
    with kpi2:
        container = st.container()#border=True)
        with container:
            st.metric("Humidity", f"{filtered_data['hum'].mean().round(2)} %")

    with kpi3:
        container = st.container()#border=True)
        with container:
            st.metric("CO2", f"{filtered_data['CO2'].mean().round(2)} ppm")                    

    tab1, tab2, tab3 = st.tabs(["Temperature", "Humidity", "CO2"])

    with tab1:
        st.markdown("### Temperature in °C")
        st.plotly_chart(plt_fig(filtered_data, "tmp"))
       
    with tab2:
        st.markdown("### Humidity in %")
        st.plotly_chart(plt_fig(filtered_data, "hum"))

    with tab3:
        st.markdown("### CO2 in ppm")
        st.plotly_chart(plt_fig(filtered_data, "CO2"))


# Gaps in the data and predictions
# tmp_tab1, hum_tab2, co2_tab3 = st.tabs(["Temperature", "Humidity", "CO2"])

# with tmp_tab1:
#     st.markdown("### Temperature in °C")
#     st.plotly_chart(plt_fig(df_hour, "tmp", "markers"))
    
# with hum_tab2:
#     st.markdown("### Humidity in %")
#     st.plotly_chart(plt_fig(df_hour, "hum", "markers"))

# with co2_tab3:
#     st.markdown("### CO2 in ppm")
#     st.plotly_chart(plt_fig(df_hour, "CO2", "markers"))


# Detailed data view
st.markdown("## Detailed Data View") 
st.dataframe(filtered_data)

# detail view max min mean 



# raum102 wirft fhler auf 
#AttributeError: 'float' object has no attribute 'round', ja obviously, weil es den Tag nicht Gibg, müssen da exceptions eingebauen