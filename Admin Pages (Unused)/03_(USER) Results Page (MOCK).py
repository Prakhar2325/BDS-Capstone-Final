import streamlit as st
from PIL import Image
import time

image = Image.open("yt_thumbnail.jpg")

progress_text = st.write("Analysis in progress. Please wait.", key="opn")
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.03)
    my_bar.progress(percent_complete + 1, text=progress_text)
st.write("Analysis Complete!")
time.sleep(1)
subheader = st.subheader("You scored")
time.sleep(0.25)
subheader.subheader("You scored.")
time.sleep(0.25)
subheader.subheader("You scored..")
time.sleep(0.25)
subheader.subheader("You scored...")

col1,col2,col3=st.columns(3)
with col2:
    st.markdown("<center><h1 style='color:rgb(227, 18, 227)'>80!</h1></center>", unsafe_allow_html=True)
    time.sleep(0.25)
    st.image(image, caption="Image Preview")
time.sleep(0.25)
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("<h3>Way to go! You scored <b style='color:lightgreen'>80%</b>!</h3>", unsafe_allow_html=True)
st.markdown("<h4>Your thumbnail is likely to attract a <b>moderate</b> amount of viewership.", unsafe_allow_html=True)

right,wrong = st.columns(2)
with right:
    with st.expander("**THINGS YOU GOT RIGHT**"):
        st.success("Face with strong emotion", icon="✅")
        st.markdown("A strong emotion like 'surprise' or 'shock' attracts more attention from the viewers.")
        st.success("Strong color composition", icon="✅")
        st.markdown("Great job! It's hard to find proper color composition in thumbnails. It really helps the picture *pop* and reel in viewers.")
with wrong:
    with st.expander("**THINS TO IMPROVE**"):
        st.error("No English text", icon="❌")
        st.markdown("Some introductory English text can help inform the viewer and immediately hook them in.")