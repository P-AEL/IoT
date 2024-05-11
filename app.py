import streamlit as st, pandas as pd, numpy as np
import plotly.graph_objects as go
from group_data import group_data
from datetime import date


# Set page config
st.set_page_config(
    page_title="Sensor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Sensor Dashboard Building A")


# Load data
data = pd.read_csv("/Users/florian/Documents/github/study/IoT/IoT/output.csv")
df_hour = group_data(data, "h")

# Sidebar
st.sidebar.header("Sensor Dashboard Building A")
input_device = st.sidebar.selectbox(label= "Select Room", options= df_hour["device_id"].unique().tolist())
input_date = st.sidebar.date_input(label= "Select Date", value= date(2022,10,10), min_value= df_hour["date_time"].min(), max_value= df_hour["date_time"].max())

# Filter data
filtered_data = df_hour[(df_hour["device_id"].astype(str) == input_device) & (df_hour["date_time"].astype(str).str.slice(0, 10).str.contains(str(input_date)))]

# Crete plot
def plot_figure(data, y):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["date_time"],
                y=data[y],
                name=y,
                mode="lines+markers"
            )
        )
        fig.update_layout(xaxis_title="Time", yaxis_title=y)
        return fig



# Columns
c1, c2 = st.columns(2)

with c1:

    st.markdown("## Pauls Tolle Graphik")

with c2:

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        container = st.container(border=True)
        with container:
            st.metric("Temperature", f"{filtered_data["tmp"].mean().round(2)} °C")
    
    with kpi2:
        container = st.container(border=True)
        with container:
            st.metric("Humidity", f"{filtered_data["hum"].mean().round(2)} %")

    with kpi3:
        container = st.container(border=True)
        with container:
            st.metric("CO2", f"{filtered_data["CO2"].mean().round(2)} ppm")

    tab1, tab2, tab3 = st.tabs(["Temperature", "Humidity", "CO2"])

    with tab1:
        st.markdown("### Temperature in °C")
        st.plotly_chart(plot_figure(filtered_data, "tmp"))

    with tab2:
        st.markdown("### Humidity in %")
        st.plotly_chart(plot_figure(filtered_data, "hum"))

    with tab3:
        st.markdown("### CO2 in ppm")
        st.plotly_chart(plot_figure(filtered_data, "CO2"))

# Detailed data view
st.markdown("## Detailed Data View") 
st.dataframe(filtered_data)