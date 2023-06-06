import streamlit as st
import io
#from io import StringIO

file_path = "C:\Capstone II\Project Stuff\capstoneApp\yt_thumbnail.jpg"
uploaded_file = st.file_uploader("Upload your thumbnail here: ", type=['png','jpg'])

if uploaded_file is not None:
    print(uploaded_file)
    with io.open(file_path, 'rb') as image_file:
        content=image_file.read()
    #print(content)
    #print(uploaded_file==content)
    st.image(uploaded_file)
    #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #print(stringio==content)
    print(uploaded_file.getvalue())