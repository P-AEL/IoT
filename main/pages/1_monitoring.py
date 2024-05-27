import streamlit as st, dataprep as dp, pandas as pd, numpy as np


st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 25px;
}
</style>
""",
    unsafe_allow_html=True,
)


# Load data
data = pd.read_csv("/Users/florian/Documents/github/study/IoT/IoT/main/output.csv")

a0 = ["hka-aqm-a017", "hka-aqm-a014"]
a1 = ["hka-aqm-a101", "hka-aqm-a102", "hka-aqm-a103", "hka-aqm-a106", "hka-aqm-a107", "hka-aqm-a108", "hka-aqm-a111", "hka-aqm-a112"]
data = data[data["device_id"].isin(a0 + a1)]  
data["device_id"] = data["device_id"].str.replace("hka-aqm-", "")

#Sidebar
st.sidebar.header("Monitoring Dashbaord for building A")
input_col = st.sidebar.selectbox(label= "Select col to display", options= ["tmp", "hum", "CO2", "VOC", "WIFI"], index= 0)
input_agg = st.sidebar.selectbox(label= "Select input_agg level", options= ["d", "w", "m", "y"], index= 0)

unit_dict = {
    "tmp": "°C",
    "hum": "%",
    "CO2": "ppm",
    "VOC": "ppb",
    "WIFI": "dBm"
} 


agg_dict = {
    'd': ('h', 'Select day', '%Y-%m-%d'),
    'w': ('d', 'Select week', '%d.%m.%Y'),
    'm': ('w', 'Select month', '%m/%Y'),
    'y': ('m', 'Select year', '%Y')
}

df = dp.group_data(data, agg_dict[input_agg][0])
df["min"] = df["date_time"].dt.minute
df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.date
df['week'] = df['date_time'].dt.to_period('W').apply(lambda r: r.start_time)
df['month'] = df['date_time'].dt.to_period('M').apply(lambda r: r.start_time)
df["year"] = df["date_time"].dt.year

if input_agg == 'd':
    unique_values = df['day'].unique()
    hour_labels = [f"{value}:00" for value in unique_values]
    selected_value = st.sidebar.selectbox('Stunde auswählen', unique_values)
    filtered_data = df[df['day'] == selected_value]

elif input_agg == 'w':
    unique_weeks = df['week'].unique()
    week_labels = [f"{week.strftime('%d.%m.%Y')} - {(week + pd.DateOffset(days=6)).strftime('%d.%m.%Y')}" for week in unique_weeks]
    selected_label = st.sidebar.selectbox('Woche auswählen', week_labels)
    selected_value = unique_weeks[week_labels.index(selected_label)]
    filtered_data = df[df['week'] == selected_value]

elif input_agg == 'm':
    unique_months = df['month'].unique()
    month_labels = [f"{month.strftime('%m/%Y')}" for month in unique_months]
    selected_label = st.sidebar.selectbox('Woche auswählen', month_labels)
    selected_value = unique_months[month_labels.index(selected_label)]
    filtered_data = df[df['month'] == selected_value]

elif input_agg == 'y':
    unique_years = df['year'].unique()
    year_labels = [pd.to_datetime(str(year), format='%Y').strftime('%Y') for year in unique_years]
    selected_label = st.sidebar.selectbox('Monat auswählen', year_labels)
    selected_value = unique_years[year_labels.index(selected_label)]
    filtered_data = df[df['year'] == selected_value]


devices = df["device_id"].unique()
cols = st.columns(len(devices) // 5)
for i in range(0, len(devices), 5):
    with cols[i//5]:
        for device in devices[i:i+5]:
            st.markdown(f"## {device}")
            df_device_dt = filtered_data[filtered_data["device_id"] == device]
            value_mean = f"{df_device_dt[input_col].mean().round(2)} {unit_dict[input_col]}" if not df_device_dt.empty else "n/a"
            value_max = f"{df_device_dt[input_col].max().round(2)} {unit_dict[input_col]}" if not df_device_dt.empty else "n/a"
            value_min = f"{df_device_dt[input_col].min().round(2)} {unit_dict[input_col]}" if not df_device_dt.empty else "n/a"
            col_mean, col_max, col_min = st.columns(3)
            with col_mean, st.container(border=True):
                st.metric("Mean", value_mean)
            with col_max, st.container(border=True):
                st.metric("Max", value_max)
            with col_min, st.container(border=True):
                st.metric("Min", value_min)
            st.plotly_chart(dp.plt_fig(df = df_device_dt, y = input_col), use_container_width=True)

st.write(filtered_data)
