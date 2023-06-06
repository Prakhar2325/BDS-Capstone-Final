import streamlit as st
from PIL import Image
import time
import json

image = Image.open('sign in with google.png')

st.title("Log Into Your Account.")

#users_dict = {'email':['prakharmishra2002@gmail.com','joshuabenzon@gmail.com','omsh97@gmail.com','bhavsh@gmail.com','dure0611@gmail.com','owensvin@gmail.com', 'gargi.asthana@gmail.com'], 'pwd':['admin','admin','admin','admin','admin','nub','nub'], 'username':['Prakhar Mishra','Joshua Benzon','Omkar Shitole','Bhavna Sharma','Duurenbat Batchuluun','Gavrielowens Vincentio','Gargi Asthana']}
#pwd_key = {'prakharmishra2002@gmail.com':'admin','joshuabenzon@gmail.com':'admin','omsh97@gmail.com':'admin','bhavsh@gmail.com':'admin','dure0611@gmail.com':'admin','owensvin@gmail.com':'nub','gargi.asthana@gmail.com':'nub'}

#usernames_key = {'prakharmishra2002@gmail.com':'Prakhar Mishra','joshuabenzon@gmail.com':'Joshua Benzon','omsh97@gmail.com':'Omkar Shitole','bhavsh@gmail.com':'Bhavna Sharma','dure0611@gmail.com':'Duurenbat Batchuluun','owensvin@gmail.com':'Gavrielowens Vincentio','gargi.asthana@gmail.com':'Gargi Asthana'}

f1 = open("users_dict.json")
users_dict = json.load(f1)
#print(data1)
#print(type(data1))

f2 = open("pwd_key.json")
pwd_key = json.load(f2)
#print(data2)
#print(type(data2))

f3 = open("usernames_key.json")
usernames_key = json.load(f3)
#print(data3)
#print(type(data3))

admins_list = ['Prakhar Mishra','Joshua Benzon','Omkar Shitole','Bhavna Sharma','Duurenbat Batchuluun']



tab1, tab2 = st.tabs(['Sign In','Sign Up'])
with tab1:
    st.markdown('<h2>Sign In</h2>', unsafe_allow_html=True)
    st.markdown('')
    email_signIn = st.text_input('Email:', placeholder='john.doe@gmail.com')
    pwd_signIn = st.text_input('Password: ', placeholder='Enter Password...', type='password')
    signIn = st.button('Sign in', type='primary', use_container_width=True)
    st.checkbox('Remember me ')

st.markdown('<hr>', unsafe_allow_html=True)
with tab2:
    st.markdown('<h2>Sign Up</h2>', unsafe_allow_html=True)
    st.markdown('✔️ Revisit feedback from your previous thumbnail reviews')
    st.markdown('✔️ Get a new YouTube thumbnail review')
    st.markdown('✔️ Receive a detailed review including areas for improvement')
    st.markdown('<br>',unsafe_allow_html=True)
    username_signUp = st.text_input('Username:',placeholder='John Doe')
    email_signUp = st.text_input('Email:', placeholder='john.doe@email.com')
    pwd_signUp = st.text_input('Password:', placeholder='Enter Password...', type='password')
    pwd_conf = st.text_input('Confirm Password:', placeholder='Confirm Password...', type='password')
    signUp = st.button('Create Account', type='primary', use_container_width=True)
    st.checkbox('Remember me')
    
if signIn:
    if email_signIn=="":
        #st.markdown("<p style='color:red'>Email field is mandatory</p>", unsafe_allow_html=True)
        st.warning("Email field is mandatory", icon="⚠")
    elif pwd_signIn=="":
        #st.markdown("<p style='color:red'>Password field is mandatory</p>", unsafe_allow_html=True)
        st.warning("Password field is mandatory", icon="⚠")
    else:
        with st.spinner("Authenticating..."):
            time.sleep(1.5)
        if email_signIn not in users_dict['email']:
            #st.markdown("<p style='color:red'>Email ID may be incorrect</p>", unsafe_allow_html=True)
            st.error("Email ID may be incorrect",icon="❌")
        elif pwd_signIn not in users_dict['pwd']:
            #st.markdown("<p style='color:red'>Password may be incorrect</p>", unsafe_allow_html=True)
            st.error("Password may be incorrect",icon="❌")
        elif pwd_key[email_signIn]!=pwd_signIn:
            st.error("Username or password may be incorrect",icon="❌")
        else:
            st.success(f"Welcome **{usernames_key[email_signIn]}**! Please navigate to 'Upload A Thumbnail' page from the sidebar to upload your desired thumbnail.", icon="✔")
            logged_in = {'email':email_signIn, 'pwd':pwd_signIn, 'username':usernames_key[email_signIn]}
            with open("users_dict.json", "w") as outfile:
                json.dump(users_dict, outfile)
    
            with open("pwd_key.json", "w") as outfile:
                json.dump(pwd_key, outfile)
    
            with open("usernames_key.json", "w") as outfile:
                json.dump(usernames_key, outfile)
                
            with open("logged_in.json", "w") as outfile:
                json.dump(logged_in, outfile)
        
if signUp:
    if username_signUp=="":
        st.warning("Please enter a user name", icon="⚠")
    elif email_signUp=="":
        #st.markdown("<p style='color:red'>Email field is mandatory</p>", unsafe_allow_html=True)
        st.warning("Email field is mandatory", icon="⚠")
    elif email_signUp in users_dict['email']:
        st.warning("Email already exists", icon="⚠")
    elif pwd_signUp=="":
        #st.markdown("<p style='color:red'>Password field is mandatory</p>", unsafe_allow_html=True)
        st.warning("Password field is mandatory", icon="⚠")
    elif pwd_conf=="":
        #st.markdown("<p style='color:red'>You must confirm your password</p>", unsafe_allow_html=True)
        st.warning("You must confirm your password", icon="⚠")
    elif pwd_signUp!=pwd_conf:
        #st.markdown("<p style='color:red'>You must enter the same password in both fields</p>", unsafe_allow_html=True)
        st.warning("You must enter the same password in both fields", icon="⚠")
    else:
        with st.spinner("Registering..."):
            time.sleep(1.5)
        users_dict['email'].append(email_signUp)
        users_dict['pwd'].append(pwd_signUp)
        users_dict['username'].append(username_signUp)
        pwd_key[email_signUp] = pwd_signUp
        usernames_key[email_signUp] = username_signUp
        st.success(f"Welcome **{usernames_key[email_signUp]}**! Please navigate to 'Upload A Thumbnail' page from the sidebar to upload your desired thumbnail.", icon="✔")
        logged_in = {'email':email_signUp, 'pwd':pwd_signUp, 'username':usernames_key[email_signUp]}
        with open("users_dict.json", "w") as outfile:
            json.dump(users_dict, outfile)
    
        with open("pwd_key.json", "w") as outfile:
            json.dump(pwd_key, outfile)
    
        with open("usernames_key.json", "w") as outfile:
            json.dump(usernames_key, outfile)
            
        with open("logged_in.json", "w") as outfile:
            json.dump(logged_in, outfile)
        

#st.write(users_dict)
#st.markdown("")
#st.write(pwd_key)
#st.markdown("")
#st.write(usernames_key)



#print(users_dict)

#col1, col2, col3 = st.columns(3)

#with col1:
#    pass
#with col2:
#    st.markdown('<center>or</center>',unsafe_allow_html=True)
#    st.markdown('')
#    st.image(image)
#with col3:
#    pass
