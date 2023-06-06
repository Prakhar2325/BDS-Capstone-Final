import streamlit as st
from PIL import Image

neon = Image.open("yt_neon.png")
nn = Image.open("nn.gif")

# Setting background image (from url):
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-photo/abstract-luxury-gradient-blue-background-smooth-dark-blue-with-black-vignette-studio-banner_1258-117928.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
#---------------------------------------------

col1, col2, col3 = st.columns(3)

with col2:
    st.image(neon)

st.markdown("<h1>Receive <b style='color:rgb(222, 24, 70)'>neural network-powered</b> feedback on your YouTube thumbnails</h1>", unsafe_allow_html=True)
st.markdown("""
In an oversaturated YouTube market, content creators need constant viewer traffic to stay afloat. Prospective viewers only have 3 previews of a creator’s video: the title, thumbnail, and video duration. As such, thumbnails are a vital part of driving traffic to video and its creator’s channel. A good thumbnail will easily attract viewers to your video, while bad thumbnails keep your potential viewers uninterested.

Our goal is to help you with the process of designing an effective, eye-catching thumbnail that will hook your viewers and grow your channel exponentially. Along with a score, YouTube Thumbnail Scorer™ will also provide you with a detailed feedback to ensure maximum improvement.
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.title("Commonly Asked Questions: ")

with st.expander("How does YouTube Thumbnail Scorer™ calculate score and provide feedback on my thumbnails?"):
    st.markdown("Our app extracts a large volume of video thumbnails and other information from the YouTube platform. It then uses Google Vision to strip each thumbnail down to its basic features and qualities. It then neatly collects all this labelled information and uses it to train a variety of models, allowing it to gradually learn the qualities of good and bad thumbnails from said labels. After rigorous training and testing, our app is effectively able to score your thumbnail and list out key areas for improvement!")
    st.image("https://i.pinimg.com/originals/b5/d3/69/b5d3692a872936d05a3d770e5327c6ec.gif")

with st.expander("Will YouTube Thumbnail Scorer™ store any of my personal data?"):
    st.markdown("Rest easy knowing that our app **will not in any capacity** store any personal or confidential data from your Google account. All the data gathered will be what's publicly available on your YouTube page and videos.")
    st.image("https://ideas.ted.com/wp-content/uploads/sites/3/2016/08/featured_art.gif")

with st.expander("Plans for the future?"):
    st.markdown("We have some interesting ideas cooking in HQ (widgets sure are nifty, aren't they?) but for the foreseeable future, this web app is the only place for you to use this technology. The more you use it, the more we expand, the cooler the updates!")
    st.image("https://cdn.vox-cdn.com/thumbor/rquCKDieH8avx-ifzKsJLV_Q6T0=/0x0:1800x1200/1800x1017/filters:focal(895x1039:896x1040):no_upscale()/cdn1.vox-cdn.com/uploads/chorus_asset/file/9139481/Casey_Youtube_GIF.gif")