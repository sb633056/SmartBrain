import streamlit as st
import requests

st.title("SmartBrain Web App")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    st.write("Uploading file to backend...")

    files = {"file": (uploaded_file.name, uploaded_file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}

    response = requests.post("http://127.0.0.1:8000/kpi/build", files=files)

    if response.status_code == 200:
        st.success("Backend response:")
        st.json(response.json())
    else:
        st.error("Something went wrong!")
