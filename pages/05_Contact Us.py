import streamlit as st
import time

# Setting a background image (from URL):


st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpaperaccess.com/full/1155017.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

st.markdown("""
<h1 style='color:rgb(176, 42, 209)'><center>GIVE US A SHOUT</center></h1>
""",unsafe_allow_html=True)
st.markdown("<center><hr width='15%' align=\"center\" style='background-color:red'></center>",unsafe_allow_html=True)
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("<h5>USE THE FORM BELOW TO DROP US AN EMAIL. OLD-FASHIONED PHONE CALLS WORK TOO >> <i style='color:rgb(209, 42, 75)'>+61 412345678</i></h5>", unsafe_allow_html=True)
dict1 = {}
with st.form("my_form"):
    col1, col2 = st.columns(2)
    with col1:
        fname = st.text_input("FName",placeholder="First Name*", label_visibility='collapsed')
        email = st.text_input("Email",placeholder="Email*", label_visibility='collapsed')
        mobile = st.text_input("Mobile",placeholder="Mobile", label_visibility='collapsed')
    with col2:
        lname = st.text_input("LName",placeholder="Last Name*", label_visibility='collapsed')
        phone = st.text_input("Phone",placeholder="Phone", label_visibility='collapsed')
        company = st.text_input("Company",placeholder="Company", label_visibility='collapsed')
    message = st.text_area("Message",placeholder="Write your message here...", label_visibility="collapsed",max_chars=4000)
    submitted = st.form_submit_button("Submit")
    if submitted:
        if fname=="":
            st.markdown("<p style='color:red'>First name is mandatory</p>", unsafe_allow_html=True)
        elif lname=="":
            st.markdown("<p style='color:red'>Last name is mandatory</p>", unsafe_allow_html=True)
        elif email=="":
            st.markdown("<p style='color:red'>Email is mandatory</p>", unsafe_allow_html=True)
        elif message=="":
            st.markdown("<p style='color:red'>Message is mandatory</p>", unsafe_allow_html=True)
        else:
            dict1['fName']=fname
            dict1['lName']=lname
            dict1['email']=email
            dict1['phone']=phone
            dict1['mobile']=mobile
            dict1['company']=company
            with st.spinner("Submitting..."):
                time.sleep(5)
                st.success("Done!", icon="✅")
                st.balloons()