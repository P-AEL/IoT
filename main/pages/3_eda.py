import streamlit as st, dataprep as dp, pandas as pd, plotly.express as px
from stoc import stoc
from copy import deepcopy

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize stoc for generating a table of contents
toc = stoc()

# Unique plot funciton for this page 
def plt_data(data: pd.DataFrame, x: str= "data_time", y: str="tmp", color: str="device_id"):
    fig = px.scatter(data, x= x, y= y, color= color)
    return fig

# Load data
FILENAME = "agg_hourly.parquet"
data = dp.load_data(FILENAME)
df = deepcopy(data)


# Sidebar
st.sidebar.header("EDA for building A")
input_use_influx_db_data = st.sidebar.checkbox(label= "Use InfluxDB data", value= False)


# Page content
toc.h1("Exploratory Data Analysis")

toc.h2("Data Overview")

toc.h3("Gaps in the data")
st.write(f"We currently have {df.shape[0]} rows and {df.shape[1]} columns in our dataset. In general we should have data from {df.date_time.min().date()} to {df.date_time.max().date()}, which is , as you can see in the following plot, not the case.")
st.plotly_chart(plt_data(df, x= "date_time", y= "tmp", color= "device_id"), use_container_width= True)
st.write(f"These gaps in the data are due to the fact that some devices in some rooms either could not uphold a stable connection to the endpoint or were turned off. The gaps are generally big but tend to differ in size and quantity per device. In the following table we can see the relative number of missing days per device and year.")

df["n_of_days"] = 0
for i in df.device_id.unique():
    df.loc[df["device_id"] == i, "n_of_days"] = df


df["date"] = df["date_time"].dt.date
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date_time"].dt.year

df_daysmissing = df.groupby(["device_id","year"]).nunique()[["date"]].reset_index()
df_daysmissing["total_days_2022"] = (pd.to_datetime("2022-12-31") - pd.to_datetime(df.date.min())).days
df_daysmissing["total_days_2023"] = (pd.to_datetime(df.date.max()) - pd.to_datetime("2023-01-01")).days
df_daysmissing["missing_days"] = 0
for i in df.year.unique():
    if i == 2022:
        df_daysmissing.loc[df_daysmissing.year == i, "missing_days"] = (df_daysmissing.total_days_2022 - df_daysmissing.date)/df_daysmissing.total_days_2022
    else:
        df_daysmissing.loc[df_daysmissing.year == i, "missing_days"] = (df_daysmissing.total_days_2023 - df_daysmissing.date)/df_daysmissing.total_days_2023

col_table, col_plt = st.columns(2)

with col_table:
    st.write(df_daysmissing)
with col_plt:
    st.plotly_chart(px.sunburst(df_daysmissing, path= ["year", "device_id"], values= "missing_days"))

device_ids = df.groupby('device_id')['year'].nunique()
device_ids = device_ids[device_ids == 1].index.tolist()

grouped_df = df.groupby(['device_id', 'date']).size().reset_index(name='hours_per_day')
average_hours_per_day = grouped_df.groupby('device_id')[['hours_per_day']].mean()
st.write(f"The devices {device_ids} have only been actively collecting data for one year")

st.plotly_chart(px.bar(average_hours_per_day, x=average_hours_per_day.index, y='hours_per_day'), use_container_width= True)

st.write(f"The lowest average hours per day is {round(average_hours_per_day.min().values[0],2)} for the room {average_hours_per_day.idxmin().values[0]}")

st.write("Datapoints with funny tmp and hum values")

col_desc, col_tmp40 = st.columns(2)
with col_desc:
    st.write(df.describe().iloc[:,:-14])
with col_tmp40:
    st.write(df.loc[df["tmp"]>40][["device_id", "date_time", "tmp", "hum"]])

toc.h2("Feature Aanalyis")

toc.h3("SNR")

st.write("As we can see, the median is -2.557895, with the 25% and 75% quantiles at -6.500000 and 1.400000, respectively. In basic terms, SNR is the difference between the desired signal and the noise floor. Also, in terms of definition, the noise floor is the spurious background transmissions that are produced by other devices or by devices that are unintentionally generating interference on a similar frequency.")
st.markdown("* **5 dB to 10 dB:** Signal-to-noise ratio is too low for a stable connection. Noise is barely distinguishable from the actual signal.")
st.markdown("* **10 dB to 15 dB:** Considered the minimally acceptable value for an unreliable connection.")
st.markdown("* **15 dB to 25 dB:** Typically considered the minimally acceptable level for poor connection quality.")
st.markdown("* **25 dB to 40 dB:** Considered good connection quality.")
st.markdown("* **41 dB or higher:** Considered excellent connection quality.")
st.write("As can be seen in our example, we have very low SNR values, indicating that the connection to other devices is very poor.")

toc.h3("RSSI")

st.write("Similar to SNR, we also observe the same with RSSI. RSSI, or 'Received Signal Strength Indicator,' is a measurement of how well your device can hear a signal from an access point or router.")
st.markdown("FRM-RSSI | RSSI Value | Connection Quality")
st.markdown("* **40 to 45** | -50 | Excellent")
st.markdown("* **30 to 40** | -60 | Very Good")
st.markdown("* **20 to 30** | -70 | Good")
st.markdown("* **20 to 30** | -80 | Low")
st.markdown("* **10 to 20** | -90 | Very Low")
st.markdown("* **0 to 10**  | -100 | No Signal")


average_rssi = df.groupby('device_id')['rssi'].mean()
st.plotly_chart(px.bar(average_rssi, x=average_rssi.index, y='rssi'), use_container_width= True)

toc.h3("CO2")
st.write("The levels of CO2 in the air and potential health problems are:")
st.markdown("* 400 ppm: average outdoor air level.")
st.markdown("* 400–1,000 ppm: typical level found in occupied spaces with good air exchange.")
st.markdown("* 1,000–2,000 ppm: level associated with complaints of drowsiness and poor air.")
st.plotly_chart(px.scatter(df.loc[df.CO2 > 1000], x='date_time', y='CO2', color='device_id'), use_container_width= True)
with st.expander("See data"):
    st.write(df.loc[df.CO2 > 1000].iloc[:,:-14])

toc.h3("VOC")
st.write("Volatile Organic Compounds (VOCs), sometimes known as chemical pollutants, are gases emitted by many of the goods we use to build and maintain our homes. Many of these pollutants are colorless and are odorless at low levels. They can be released into the environment during the use as well as storage of products. While products emit VOCs, the amount tends to decrease with age.")
st.markdown("VOC Level | Health Effects")
st.markdown("* 0 to 400 ppb: This is the acceptable level of VOC indoors. You should not expect short-term effects such as irritation or discomfort.")
st.markdown("* 400 to 2,200 ppb: Short-term exposure can result in noticeable effects such as headaches, nausea, dizziness, and irritation of the respiratory tract and eyes.")
st.plotly_chart(px.scatter(df.loc[df.VOC > 1000], x='date_time', y='VOC', color='device_id'), use_container_width= True)
with st.expander("See data"):
    st.write(df.loc[df.VOC > 1000].iloc[:,:-14])


toc.h3("BLE")
st.plotly_chart(px.scatter(df.loc[df.BLE > 100], x='date_time', y='BLE', color='device_id'), use_container_width= True)
with st.expander("See data"):
    st.write(df.loc[df.BLE > 100].iloc[:,:-14])


st.write("All points in time coincide either with the end or the beginning of the semester. Regarding the rooms, it must be mentioned:")
st.markdown("* Room 101 is a large lecture hall/auditorium")
st.markdown("* Room 102 is the West learning room/quiet area")
st.markdown("* Room 103 is the library")

toc.h2("Correlation analysis of target variables")
df["shifted_Target"] = 0
for i in df.device_id.unique():
    df.loc[df.device_id == i, "shifted_Target"] = df.loc[df.device_id == i, "tmp"].shift(-1)
df.dropna(inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt

correlation = df.drop("device_id",axis=1).corr()['tmp']
plt.figure(figsize=(12, 8))
sns.heatmap(correlation.to_frame(), annot=True, cmap='coolwarm', cbar=False)
plt.title('Correlation between tmp and other columns')

st.pyplot(plt)

toc.toc()