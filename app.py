import os


import streamlit as st 
from Estimator import WeightEstimator

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

file = st.file_uploader(
    "Upload your data file",
    accept_multiple_files=False,
    key=st.session_state["file_uploader_key"],
)

if file:
    st.session_state["uploaded_files"] = file

if file is not None:
    path = None
    bytes_data = file.read()  # read the content of the file in binary
    with open(os.path.join("tmp", file.name), "wb") as f:
        f.write(bytes_data)  # write this content elsewhere
    path = os.path.join("tmp", file.name)
    weight,im,gr,thr,CImg,label = WeightEstimator(path)
    st.header("Original image")
    st.image(im)
    st.header("Gray Scale image")
    st.image(gr)
    st.header("Thresholded image")
    st.image(thr)
    st.header("Edge detected image")
    st.image(CImg)
    st.header("Predicted Class")
    st.write(label)
    st.header("Estimated weight")
    st.write(weight)