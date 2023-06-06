import streamlit as st
from PIL import Image
logo = Image.open("YouTube Thumbnail Analysis3.png")
result = Image.open("YouTube Thumbnail Scorer Analysis Page Mockup.png")

# Setting background image (from URL):
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpapers.com/images/hd/dark-gradient-rpa97mj9dhic8raf.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
#---------------------------------------------

col1, col2, col3 = st.columns([1,0.1,27])

with col1:
    st.image(logo, width=37)
with col3:
    st.markdown("<h6 style='color:rgb(227, 59, 48,0.9);'>YouTube Thumbnail Scorer</h6>", unsafe_allow_html=True)

st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
col4, col5 = st.columns(2, gap="large")
with col4:
    st.markdown("<h3 style='color:rgb(75, 70, 240)'><b>Improve your YouTube video thumbnails</b></h3>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("<h6 style='color:rgb(155, 155, 158)'>Designed by our top recruiters, our AI powered platform instantly gives you tailored feedback on your desired YouTube video thumbnail.</h6>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    col6, col7 = st.columns(2)
    with col6:
        start = st.button("GET STARTED", type="primary", use_container_width=True)
            
    with col7:
        learn = st.button("LEARN MORE", use_container_width=True)
            
with col5:
    st.image(result)
    
if start:
    st.markdown("Navigate over to the sidebar to **log in**!")
if learn:
    st.markdown("Navigate over to the sidebar to learn about our **features, developers and more**!")