import streamlit as st, dataprep as dp, pandas as pd



st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
data = pd.read_csv("/Users/florian/Documents/github/study/IoT/IoT/main/output.csv")

a0 = ["hka-aqm-a017", "hka-aqm-a014"]
a1 = ["hka-aqm-a101", "hka-aqm-a102", "hka-aqm-a103", "hka-aqm-a106", "hka-aqm-a107", "hka-aqm-a108", "hka-aqm-a111", "hka-aqm-a112"]
data = data[data["device_id"].isin(a0 + a1)]  
data["device_id"] = data["device_id"].str.replace("hka-aqm-", "")


st.plotly_chart(dp.plt_fig(data, "date_time", "tmp"))

st.markdown("#### Diese Lücken bestehen als größere Lücken generel und auch zwischen einzelnen Räumen, siehe Vergleich Raum 017")

