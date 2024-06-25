import streamlit as st, pandas as pd, pyvista as pv, dataprep as dp
from stpyvista import stpyvista
from datetime import date
from pyvista_test import plot_cube
from copy import deepcopy
import os 
import logging
import influxdb_client
import json
logging.basicConfig(level=logging.INFO)


FILEPATH_INFLUXDB_CREDENTIALS = ""

# Page config
st.set_page_config(
    page_title="Sensor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Sidebar
st.sidebar.header("Sensor Dashboard Building A")
input_device = st.sidebar.selectbox(label= "Select Room", options= df["device_id"].unique().tolist() + ["all"], index= 2)
input_date = st.sidebar.date_input(label= "Select Date", value= date(2022,10,10), min_value= df["date_time"].min(), max_value= df["date_time"].max())

# Use the session state value as the default value for the checkbox
st.session_state["influxdb"] = st.sidebar.checkbox(label="Use InfluxDB data", value= st.session_state["influxdb"])
#st.session_state["influxdb"] = input_use_influx_db_data

# Update the session state value if the checkbox value changes
if st.session_state["influxdb"]:
    st.sidebar.write("InfluxDB data is used.")
else:
    st.sidebar.write("InfluxDB data is not used.")

# Load data
FILENAME = "agg_hourly.parquet"
data = load_data(FILENAME, use_influx_db= st.session_state["influxdb"])
df = deepcopy(data)


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

