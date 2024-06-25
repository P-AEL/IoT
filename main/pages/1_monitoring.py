import streamlit as st, dataprep as dp, pandas as pd
import os
import logging
import influxdb_client
import json
logging.basicConfig(level=logging.INFO)

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

FILEPATH_INFLUXDB_CREDENTIALS = ""

# Functions
def read_credentials(
        filename: str= "credentials.json"
        ) -> dict:
    """
    Reads the credentials from a given filename.

    returns: dict
    """
    try:
        with open(filename, "r") as file:
            credentials = json.load(file)

    except FileNotFoundError:
        logging.error(f"Could not find file {filename}.")
        raise 
    
    return credentials

def make_tz_naive(
        timestamp
    ):
    """
    Makes a timestamp timezone naive.

    returns: timestamp
    """
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert(None).tz_localize(None)

    return timestamp

def extract_data_from_influxdb(
        credentials: dict,
        query: str,
    ) -> list:
    """
    Create client and instantiate read/write API

    returns: list
    """
    write_client = influxdb_client.InfluxDBClient(url= credentials['url'], token= credentials['token'], org= credentials['org'])
    query_api = write_client.query_api()
    tables = query_api.query(query, org= credentials['org'])

    data = []
    for table in tables:
        for record in table.records:
            time = record.get_time()
            value = record.get_value()
            measurement = record.values.get('_measurement')
            field = record.values.get('_field')
            sensor = record.values.get('sensor')
            data.append([time, value, measurement, field, sensor])

    return data

@st.cache_data
def load_data(
        filename: str= "agg_hourly.parquet",
        use_influx_db: bool= False
    ) -> pd.DataFrame:
    """
    Loads data from given filename or InfluxDB.
    
    returns: pd.DataFrame
    raises: FileNotFoundError if not able to load the file.
    """
    df_new = []

    if use_influx_db:
        # Get credentials for InfluxDB
        credentials = read_credentials(FILEPATH_INFLUXDB_CREDENTIALS)
        query = f"""from(bucket: "{credentials['bucket']}")
        |> range(start: 2021-03-17T23:30:00Z)"""

        data = extract_data_from_influxdb(credentials, query)

        # Define column names
        column_names = ['date_time', 'value', 'measurement', 'field', 'sensor']

        # Create the DataFrame
        df = pd.DataFrame(data, columns=column_names)
        df_new = []
        fields = ['tmp', 'CO2', 'BLE', 'IR', 'WIFI', 'VOC', 'bandwidth', 'channel_index', 'channel_rssi', 'f_cnt', 'hum', 'rssi', 'spreading_factor', 'vis', 'device_id', 'snr', 'gateway']
        for time in df['date_time'].unique():
            df_time = df.loc[df['date_time'] == time]
            row = [time]
            for field in fields:        
                row.append(df_time[df_time['measurement'].str.contains(field)]['value'].tolist()[0])
            df_new.append(row)

        df_new_room_014 = pd.DataFrame(df_new, columns=['date_time'] + fields)
        df_new_room_014['date_time'] = df_new_room_014['date_time'].apply(make_tz_naive)

        try:
            filepath = os.path.join("./data/aggregated_data/", filename)
            df_new_without_room_014 = pd.read_parquet(filepath)

        except FileNotFoundError:
            logging.error(f"File {filepath} does not exist.")
            raise

        df_new_without_room_014 = df_new_without_room_014[df_new_without_room_014['device_id'] != 'a014']
        df_new = pd.concat([df_new_room_014, df_new_without_room_014], ignore_index=True)
        df_new.date_time = pd.to_datetime(df_new.date_time)

        if filename == "agg_hourly.parquet":
            df_new = dp.group_data(df_new, freq="h") 

    try:
        filepath = os.path.join("./data/aggregated_data/", filename)
        df_new = pd.read_parquet(filepath)

    except FileNotFoundError:
        logging.error(f"File {filepath} does not exist.")
        raise

    return df_new


# Initialize session state
if "influxdb" not in st.session_state:
    st.session_state["influxdb"] = False

# Use the session state value as the default value for the checkbox
st.session_state["influxdb"] = st.sidebar.checkbox(label="Use InfluxDB data", value= st.session_state["influxdb"])
#st.session_state["influxdb"] = input_use_influx_db_data

# Update the session state value if the checkbox value changes
if st.session_state["influxdb"]:
    st.sidebar.write("InfluxDB data is used.")
else:
    st.sidebar.write("InfluxDB data is not used.")

# Sidebar
st.sidebar.header("Monitoring Dashbaord for building A")
input_col = st.sidebar.selectbox(label= "Select col to display", options= ["tmp", "hum", "CO2", "VOC", "WIFI"], index= 0)
input_agg = st.sidebar.selectbox(label= "Select input_agg level", options= ["d", "w", "m", "y"], index= 0)

# Load data
FILENAME = "output.parquet"
data = load_data(FILENAME, use_influx_db = st.session_state["influxdb"])

unit_dict = {
    "tmp": "°C",
    "hum": "%",
    "CO2": "ppm",
    "VOC": "ppb",
    "WIFI": "dBm"
} 

agg_dict = {
    "d": ("h", "Select day", "%Y-%m-%d"),
    "w": ("d", "Select week", "%d.%m.%Y"),
    "m": ("w", "Select month", "%m/%Y"),
    "y": ("m", "Select year", "%Y")
}

# Page content
df = dp.group_data(data, agg_dict[input_agg][0])
df["min"] = df["date_time"].dt.minute
df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.date
df["week"] = df["date_time"].dt.to_period("W").apply(lambda r: r.start_time)
df["month"] = df["date_time"].dt.to_period("M").apply(lambda r: r.start_time)
df["year"] = df["date_time"].dt.year

if input_agg == "d":
    unique_values = df["day"].unique()
    hour_labels = [f"{value}:00" for value in unique_values]
    selected_value = st.sidebar.selectbox("Select hour", unique_values)
    filtered_data = df[df["day"] == selected_value]

elif input_agg == "w":
    unique_weeks = df["week"].unique()
    week_labels = [f'{week.strftime("%d.%m.%Y")} - {(week + pd.DateOffset(days=6)).strftime("%d.%m.%Y")}' for week in unique_weeks]
    selected_label = st.sidebar.selectbox("Select week", week_labels)
    selected_value = unique_weeks[week_labels.index(selected_label)]
    filtered_data = df[df["week"] == selected_value]

elif input_agg == "m":
    unique_months = df["month"].unique()
    month_labels = [f'{month.strftime("%m/%Y")}' for month in unique_months]
    selected_label = st.sidebar.selectbox("Select month", month_labels)
    selected_value = unique_months[month_labels.index(selected_label)]
    filtered_data = df[df["month"] == selected_value]

elif input_agg == "y":
    unique_years = df["year"].unique()
    year_labels = [pd.to_datetime(str(year), format="%Y").strftime("%Y") for year in unique_years]
    selected_label = st.sidebar.selectbox("Select year", year_labels)
    selected_value = unique_years[year_labels.index(selected_label)]
    filtered_data = df[df["year"] == selected_value]


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
