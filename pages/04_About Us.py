import streamlit as st
from PIL import Image

josh = Image.open("capstone josh.png")
omkar = Image.open("capstone omkar.png")
bhavna = Image.open("capstone bhavna.png")
prakhar = Image.open("capstone prakhar.png")
dure = Image.open("capstone dure.png")

# Setting background image (from url):
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1557682224-5b8590cd9ec5?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1yZWxhdGVkfDE0fHx8ZW58MHx8fHw%3D&w=1000&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

st.markdown("""
<h1 style='color:rgb(19, 227, 11)'><center>MEET THE TEAM</center></h1>
""",unsafe_allow_html=True)
#st.markdown("")
#st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
col1,col2,col3,col4,col5 = st.columns(5)
with col1:
    st.image(josh)
    st.markdown("<center><b style='color:rgb(57, 58, 59)'>Joshua Benzon</b></center>", unsafe_allow_html=True)
    st.markdown("<center> <strike>Mad</strike> Data scientist</center>", unsafe_allow_html=True)
with col2:
    st.image(omkar)
    st.markdown("<center><b style='color:rgb(57, 58, 59)'>Omkar Shitole</b></center>", unsafe_allow_html=True)
    st.markdown("<center>Nocturnal coder</center>", unsafe_allow_html=True)
with col3:
    st.image(bhavna)
    st.markdown("<center><b style='color:rgb(57, 58, 59)'>Bhavna Sharma</b></center>", unsafe_allow_html=True)
    st.markdown("<center>Writer, designer, heavy sleeper</center>", unsafe_allow_html=True)
with col4:
    st.image(prakhar)
    st.markdown("<center><b style='color:rgb(57, 58, 59)'>Prakhar Mishra</b></center>", unsafe_allow_html=True)
    st.markdown("<center>UI and ice-cream enthusiast</center>", unsafe_allow_html=True)
with col5:
    st.image(dure)
    st.markdown("<center><b style='color:rgb(57, 58, 59)'>Duurenbat Batchuluun</b></center>", unsafe_allow_html=True)
    st.markdown("<center>Good taste in music</center>", unsafe_allow_html=True)
    
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("")
st.markdown("")
st.markdown("<h1>Growing Up With YouTube</h1>", unsafe_allow_html=True)
st.image("https://cliply.co/wp-content/uploads/2021/09/CLIPLY_142110380_ORGANIC_YT_ICON_400.gif", width=200)
st.markdown("Our humble team of five grew up watching YouTube, like most others our generation. It has had a huge impact on each of our lives. Some of us even dabbled in it ourselves (best not to look)! We understand the platform and its stars through and through, and we haven't ignored the myriad of challenges that so many YouTubers have faced over the years. Our love for the platform and its people is what brought us together and inspired us to create this app.", unsafe_allow_html=True)
#col6,col7 = st.columns(2)
#with col6:
#    st.markdown("Our humble team of five grew up watching YouTube, like most others our generation. It has had a huge impact on each of our lives. Some of us even dabbled in it ourselves (best not to look)! We understand the platform and its stars through and through, and we haven't ignored the myriad of challenges that so many YouTubers have faced over the years. Our love for the platform and its people is what brought us together and inspired us to create this app.", unsafe_allow_html=True)
#with col7:
#    st.image("https://media.tenor.com/IIXy6CqR5l8AAAAC/mrbeast-ytpmv.gif")
    
#col8,col9 = st.columns(2)

#st.markdown("<h1>Our Mission<h1>", unsafe_allow_html=True)
#with col8:
#    st.markdown("What we've set out to do is pretty simple: With the heavy competition present in the market right now, we want up and coming YouTubers to get a headstart, rather than floundering for months or years on end. Our app takes what is arguably the most important component you need to market your videos - i.e. the <b>thumbnail</b> - and scores it, gives you integral feedback. The thumbnail is always the first thing potential viewers (and subscribers, fingers crossed) will see; and with people having the attention span of a butterfly, you need to attract them to your videos immediately! With this tool, we hope you", unsafe_allow_html=True)
#with col9:
st.markdown("<h1>Our Mission<h1>", unsafe_allow_html=True)
st.image("https://media.tenor.com/IIXy6CqR5l8AAAAC/mrbeast-ytpmv.gif", width=250)
st.markdown("What we've set out to do is pretty simple: with the heavy competition present in the market right now, we want up-and-coming YouTubers to get a headstart, rather than floundering for months or years on end. Our app takes what is arguably the most important component you need to market your videos - i.e. the <b>thumbnail</b> - and scores it, gives you integral feedback. The thumbnail is always the first thing potential viewers (and subscribers, fingers crossed) will see; and with people having the attention span of a butterfly, you need to attract them to your videos immediately! With this tool, we are sure you'll succeed.", unsafe_allow_html=True)
st.markdown("")
st.markdown("")
st.markdown("<center><h3>If you wish to <b style='color:rgb(4, 81, 214)'>contact us</b> or learn more about the <b style='color:rgb(19, 235, 130)'>features</b> of this app, navigate to the sidebar to learn more!", unsafe_allow_html=True)