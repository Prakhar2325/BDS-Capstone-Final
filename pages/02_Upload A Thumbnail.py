# UI Libraries
import streamlit as st
import streamlit.components.v1 as components
# File Manipulation Libraries
import io
import os
import pandas as pd
from PIL import Image
import json
# Google Libraries
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from google.oauth2 import service_account
# ML Libraries
from xgboost import XGBClassifier
from ast import literal_eval
# Miscellaneous Libraries
import numpy as np
from tqdm import tqdm
import base64
import time

# Setting background image (from LOCAL):

#with open('background.jpg', 'rb') as image_file:
    #encoded_string = base64.b64encode(image_file.read())
#st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
#             background-size: cover
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
#---------------------------------------------
try:
    f = open("logged_in.json")
    logged_in = json.load(f)
    #f.close()
    st.sidebar.markdown(f"Logged in as **{logged_in['username']}**")
    signOut = st.sidebar.button("Sign Out")
    if signOut:
        os.remove("logged_in.json")
    #----------------------------------------------------------------
    yt = Image.open('yt_play.png')
    col1, col2 = st.columns(2)

    with col1:
        st.title("Upload a Thumbnail.")
        st.markdown("Without a good thumbnail, you won't be able to maximise your video's potential.")
    with col2:
        uploaded_file = st.file_uploader("Upload your thumbnail here: ", type=['png','jpg'])
        #if uploaded_file is not None:
            #st.image(uploaded_file, caption='Image Preview')

    st.markdown("<hr>", unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col4:
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Image Preview')
            st.write("Upload?")
            confirm = st.button("CONFIRM",use_container_width=True)
            #confirm = st.button("CONFIRM",use_container_width=True)
            if confirm:
                with st.spinner("Uploading..."):
                    time.sleep(1.5)
                    st.success("Done!")

    #IMAGE_PATH = input('Path to image: ')
    if uploaded_file is not None:
        st.markdown("<hr>",unsafe_allow_html=True)
        with tqdm(total = 12) as pbar:
            ### ████████████████████████████████████████████████████████ 
            ### #################### EXTERNAL SETUP ####################
            ### ████████████████████████████████████████████████████████

            pbar.set_description_str("Authenticating through Google...")
            authenticating = st.markdown("Authenticating through Google.")
            time.sleep(0.3)
            authenticating.markdown("Authenticating through Google..")
            time.sleep(0.3)
            authenticating.markdown("Authenticating through Google...")
            time.sleep(0.3)

            credentials = service_account.Credentials.from_service_account_file(
                'yt-thumbnail-scorer-b9db9cf33c74.json'
                )

            client = vision_v1.ImageAnnotatorClient(credentials=credentials)

            pbar.update(1)
            pbar.set_description_str('Loading embedder...')

            embeddings_dict = {}
            with open("glove.twitter.27B.25d.txt", 'r', encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    embeddings_dict[word] = vector

            del embeddings_dict['0.065581'] # this one has 24 numbers in its vector for some reason

            embed_keys = embeddings_dict.keys()

            pbar.update(1)
            pbar.set_description_str('Loading model...')
            authenticating.markdown("Done!")
            loading = st.markdown("Loading the model.")
            time.sleep(0.3)
            loading.markdown("Loading the model..")
            time.sleep(0.3)
            loading.markdown("Loading the model...")
            time.sleep(0.3)

            model = XGBClassifier()
            model.load_model('yt_scorer.json')

            ### ████████████████████████████████████████████████████████ 
            ### ############## FEATURE EXTRACTION METHODS ##############
            ### ████████████████████████████████████████████████████████

            def strings_to_objects(df):
                literal_columns = ['faces', 'labels', 'objects', 'text']
                for i in literal_columns:
                    df[i] = df[i].apply(literal_eval)
                df['dominant_colors'] = df['dominant_colors'].apply(lambda x: split_colors(x))
                for i in ['adult', 'spoof', 'medical', 'violence', 'racy']:
                    df[i] = df[i].apply(lambda x: x.split('.')[1])
                return df

            def split_colors(x):
                try:
                    colors = []
                    for i in x[1:-1].split(', '):
                        this_color = []
                        for j in i.split('\n')[:3]:
                            if j:
                                this_color.append(int(float(j.split(': ')[1])))
                        if len(this_color) == 3:
                            colors.append(this_color)
                    return colors
                except:
                    print('ERROR')
                    print('original')
                    #print(x)
                    for n, i in enumerate(x[1:-1].split(', ')):
                        #print(n, i)
                        for j in i.split('\n')[:3]:
                            if len(j.split(': ')) < 2: 
                                print(x is True)
                                print(j)
                                print(j.split(': '))
                            #print("--", int(float(j.split(': ')[1])))

            data = []
            def extract_features(row):
                likelihoods = {
                    'VERY_UNLIKELY': 0.01,
                    'UNLIKELY': 0.25,
                    'POSSIBLE': 0.5,
                    'LIKELY': 0.75,
                    'VERY_LIKELY': 1 
                }


                face = np.array([list(map(lambda x: likelihoods[x], i.values())) for i in row['faces']]).T
                #nouns = row['labels']
                colors = row['dominant_colors']

                features = {
                    'Number of faces': len(face[0]) if face.any() else 0,
                    'Face_joyLikelihood': np.mean(face[0]) if face.any() else 0,
                    'Face_sorrowLikelihood':  np.mean(face[1]) if face.any() else 0,
                    'Face_angerLikelihood':  np.mean(face[2]) if face.any() else 0,
                    'Face_underExposedLikelihood':  np.mean(face[3]) if face.any() else 0,
                    'Face_surpriseLikelihood':  np.mean(face[4]) if face.any() else 0,
                    'Face_blurredLikelihood':  np.mean(face[5]) if face.any() else 0,
                    'Face_headwearLikelihood':  np.mean(face[6]) if face.any() else 0,
                    #'Face2_joyLikelihood': face[1][0] if 1 < len(face) else 0,
                    #'Face2_sorrowLikelihood': face[1][1] if 1 < len(face) else 0,
                    #'Face2_angerLikelihood': face[1][2] if 1 < len(face) else 0,
                    #'Face2_underExposedLikelihood': face[1][3] if 1 < len(face) else 0,
                    #'Face2_surpriseLikelihood': face[1][4] if 1 < len(face) else 0,
                    #'Face2_blurredLikelihood': face[1][5] if 1 < len(face) else 0,
                    #'Face2_headwearLikelihood': face[1][6] if 1 < len(face) else 0,
                    #'Noun1_trend': nouns[0] if 0 < len(nouns) else 0,
                    #'Noun2_trend': nouns[1] if 1 < len(nouns) else 0,
                    #'Noun3_trend': nouns[2] if 2 < len(nouns) else 0,
                    #'Noun4_trend': nouns[3] if 3 < len(nouns) else 0,
                    #'Noun5_trend': nouns[4] if 4 < len(nouns) else 0,
                    'Color1_r': colors[0][0] if 0 < len(colors) else 0,
                    'Color1_g': colors[0][1] if 0 < len(colors) else 0,
                    'Color1_b': colors[0][2] if 0 < len(colors) else 0,
                    'Color2_r': colors[1][0] if 1 < len(colors) else 0,
                    'Color2_g': colors[1][1] if 1 < len(colors) else 0,
                    'Color2_b': colors[1][2] if 1 < len(colors) else 0,
                    'Color3_r': colors[2][0] if 2 < len(colors) else 0,
                    'Color3_g': colors[2][1] if 2 < len(colors) else 0,
                    'Color3_b': colors[2][2] if 2 < len(colors) else 0,
                    'adult_Likelihood': likelihoods[row['adult']],
                    'medical_Likelihood': likelihoods[row['medical']],
                    'racy_Likelihood': likelihoods[row['racy']],
                    'spoof_Likelihood': likelihoods[row['spoof']],
                    'violence_Likelihood': likelihoods[row['violence']],
                }

                labels = row['labels']
                labels = [i.lower() for i in labels]
                counter = 0
                for i in labels:
                    if i not in embeddings_dict.keys(): continue
                    counter += 1
                    for m, j in enumerate(embeddings_dict[i]):
                        features[f'Label{counter}_d{m+1}'] = j
                    if counter >= 3: break
                while counter < 3:
                    counter += 1
                    for m in range(25):
                        features[f'Label{counter}_d{m+1}'] = 0

                data.append(features)

            ### ████████████████████████████████████████████████████████ 
            ### ################## FEATURE EXTRACTION ##################
            ### ████████████████████████████████████████████████████████

            pbar.update(4)
            pbar.set_description_str("Extracting face...")
            loading.markdown("Done!")
            extractingFace = st.markdown("Extracting face.")
            time.sleep(0.15)
            extractingFace.markdown("Extracting face..")
            time.sleep(0.15)
            extractingFace.markdown("Extracting face...")
            time.sleep(0.15)

            # Loads the image into memory
            #file_path = IMAGE_PATH
            #with io.open(file_path, 'rb') as image_file:
                #content = image_file.read()
            content = uploaded_file.getvalue()

            image = types.Image(content=content)

            # Perform face detection
            response_faces = client.face_detection(image=image)
            faces = response_faces.face_annotations

            face_list = []
            likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
            for face in faces:
                face_dict = {
                    'joy_likelihood': likelihood_name[face.joy_likelihood],
                    'sorrow_likelihood': likelihood_name[face.sorrow_likelihood],
                    'anger_likelihood': likelihood_name[face.anger_likelihood],
                    'surprise_likelihood': likelihood_name[face.surprise_likelihood],
                    'under_exposed_likelihood': likelihood_name[face.under_exposed_likelihood],
                    'blurred_likelihood': likelihood_name[face.blurred_likelihood],
                    'headwear_likelihood': likelihood_name[face.headwear_likelihood],
                }
                face_list.append(face_dict)

            pbar.update(1)
            pbar.set_description_str("Extracting text...")
            extractingText = st.markdown("Extracting text.")
            time.sleep(0.15)
            extractingText.markdown("Extracting text..")
            time.sleep(0.15)
            extractingText.markdown("Extracting text...")
            time.sleep(0.15)

            # Perform text detection
            response_text = client.text_detection(image=image)
            texts = response_text.text_annotations

            pbar.update(1)
            pbar.set_description_str("Extracting image properties...")
            extractingImg = st.markdown("Extracting image properties.")
            time.sleep(0.15)
            extractingImg.markdown("Extracting image properties..")
            time.sleep(0.15)
            extractingImg.markdown("Extracting image properties...")
            time.sleep(0.15)

            # Perform image properties detection
            response_props = client.image_properties(image=image)
            props = response_props.image_properties_annotation

            pbar.update(1)
            pbar.set_description_str("Extracting labels and objects...")
            extractingObj = st.markdown("Extracting labels and objects.")
            time.sleep(0.15)
            extractingObj.markdown("Extracting labels and objects..")
            time.sleep(0.15)
            extractingObj.markdown("Extracting labels and objects...")
            time.sleep(0.15)

            # Perform label detection
            response_labels = client.label_detection(image=image)
            labels = response_labels.label_annotations

            # Perform object detection
            response_objects = client.object_localization(image=image)
            objects = response_objects.localized_object_annotations

            pbar.update(1)
            pbar.set_description_str("Extracting safe search features...")
            extractingSS = st.markdown("Extracting safe search features.")
            time.sleep(0.15)
            extractingSS.markdown("Extracting safe search features..")
            time.sleep(0.15)
            extractingSS.markdown("Extracting safe search features...")
            time.sleep(0.15)

            # Perform safe search detection
            response_safe_search = client.safe_search_detection(image=image)
            safe_search = response_safe_search.safe_search_annotation

            # Append the results to the list
            features = {
                'file_name': uploaded_file.name,
                'faces': str(face_list[:2]),
                'labels': str([label.description for label in labels]),
                'objects': str([obj.name for obj in objects]),
                'dominant_colors': str([color.color for color in props.dominant_colors.colors]),
                'adult': str(safe_search.adult),
                'spoof': str(safe_search.spoof),
                'medical': str(safe_search.medical),
                'violence': str(safe_search.violence),
                'racy': str(safe_search.racy),
                'text': str([text.description for text in texts]),
            }

            pbar.update(1)
            pbar.set_description_str("Feature engineering...")
            st.markdown("All extraction done!")
            time.sleep(0.3)
            featureEng = st.markdown("Feature engineering.")
            time.sleep(0.3)
            featureEng.markdown("Feature engineering..")
            time.sleep(0.3)
            featureEng.markdown("Feature engineering...")
            time.sleep(0.3)

            single = pd.DataFrame(features, index=[0])
            single = strings_to_objects(single)
            data = []
            single.apply(extract_features, axis=1)
            single = pd.DataFrame(data)

            ### ████████████████████████████████████████████████████████ 
            ### ###################### PREDICTION ######################
            ### ████████████████████████████████████████████████████████

            pbar.update(1)
            pbar.set_description_str("Scoring...")
            featureEng.markdown("Done!")
            time.sleep(1)
            scoretxt = st.subheader("You scored.")
            time.sleep(0.3)
            scoretxt.subheader("You scored..")
            time.sleep(0.3)
            scoretxt.subheader("You scored...")
            time.sleep(1)

            bad, good = model.predict_proba(single)[0]

        print("Rating: %.2f%%" % (good * 100.0))
        #st.write("Rating: ", good*100.0)
        perc = round(good*100, 2)
        #perc=36.62
        scorefin = st.title(f"{perc}% !")
        time.sleep(1.5)
        if perc>=90:
            st.balloons()
            st.subheader(f"Incredible! You scored {perc}% !")
            st.markdown("<h4 style='color:rgb(50, 168, 82)'>Your thumbnail is likely to attract a <b>large</b> amount of viewership!</h4>", unsafe_allow_html=True)
            img1,img2,img3 = st.columns(3)
            with img2:
                st.image("https://media.newyorker.com/photos/631113f080212a2c5be0cc79/4:3/w_2132,h_1599,c_limit/Lozano_Final.gif")
            right,wrong = st.columns(2)
            with right:
                with st.expander("**THINGS YOU GOT RIGHT**"):
                    st.success("Face with strong emotion", icon="✅")
                    st.markdown("A strong emotion like 'surprise' or 'shock' attracts more attention from the viewers.")
                    st.success("Strong color composition", icon="✅")
                    st.markdown("Great job! It's hard to find proper color composition in thumbnails. It really helps the picture *pop* and reel in viewers.")
                    st.success("Informative text", icon="✅")
                    st.markdown("Your thumbnail contains text that immediately informs the viewers about your content. This is great for hooking in your target viewers.")
            with wrong:
                with st.expander("**THINGS TO IMPROVE**"):
                    st.success("Nothing noteworthy. Keep up the good work!")
        elif perc>=70 and perc<90:
            st.balloons()
            st.subheader(f"Way to go! You scored {perc}% .")
            st.markdown("<h4 style='color:rgb(94, 194, 37)'>Your thumbnail is likely to attract a <b>significant</b> amount of viewership.</h4>",unsafe_allow_html=True)
            img4,img5,img6 = st.columns(3)
            with img5:
                st.image("https://miro.medium.com/v2/resize:fit:1026/0*RjycuvCHOmNiI8fd.gif")
            right,wrong = st.columns(2)
            with right:
                with st.expander("**THINGS YOU GOT RIGHT**"):
                    st.success("Face with strong emotion", icon="✅")
                    st.markdown("A strong emotion like 'surprise' or 'shock' attracts more attention from the viewers.")
                    st.success("Informative text", icon="✅")
                    st.markdown("Your thumbnail contains text that immediately informs the viewers about your content. This is great for hooking in your target viewers.")
            with wrong:
                with st.expander("**THINGS TO IMPROVE**"):
                    st.error("Poor color composition", icon="❌")
                    st.markdown("It's one of the most common pitfalls of thumbnails. Don't worry! A bit of brainstorming, a bit of workshopping, and your thumbnail will look its vibrant self in no time!")
        elif perc>=50 and perc<70:
            st.subheader(f"Decent work! You scored {perc}% .")
            st.markdown("<h4 style='color:rgb(201, 199, 46)'>Your thumbnail is likely to attract a <b>moderate</b> amount of viewership.</h4>", unsafe_allow_html=True)
            img7,img8,img9 = st.columns(3)
            with img8:
                st.image("https://cdn.theatlantic.com/thumbor/DpbNyojTZGag6SHkepIJwZ8Ahkg=/0x0:2000x1125/960x540/media/img/mt/2022/09/Coolhunting_Anim/original.gif")
            right,wrong = st.columns(2)
            with right:
                with st.expander("**THINGS YOU GOT RIGHT**"):
                    st.success("Face with strong emotion", icon="✅")
                    st.markdown("A strong emotion like 'surprise' or 'shock' attracts more attention from the viewers.")
            with wrong:
                with st.expander("**THINGS TO IMPROVE**"):
                    st.error("Poor color composition", icon="❌")
                    st.markdown("It's one of the most common pitfalls of thumbnails. Don't worry! A bit of brainstorming, a bit of workshopping, and your thumbnail will look its vibrant self in no time!")
                    st.error("No English text", icon="❌")
                    st.markdown("Some introductory English text can help inform the viewer and immediately hook them in.")
        elif perc>=10 and perc<50:
            st.subheader(f"Oh no! You scored {perc}% .")
            st.markdown("<h4 style='color:rgb(199, 44, 62)'>Your thumbnail is likely to attract a <b>poor</b> amount of viewership.</h4>", unsafe_allow_html=True)
            imag1,imag2,imag3 = st.columns(3)
            with imag2:
                st.image("https://media.tenor.com/_zWA5jjcC5gAAAAS/bob-the-tomato-flip.gif")
            right,wrong = st.columns(2)
            with right:
                with st.expander("**THINGS YOU GOT RIGHT**"):
                    st.markdown("")
            with wrong:
                with st.expander("**THINGS TO IMPROVE**"):
                    st.error("Poor color composition", icon="❌")
                    st.markdown("It's one of the most common pitfalls of thumbnails. Don't worry! A bit of brainstorming, a bit of workshopping, and your thumbnail will look its vibrant self in no time!")
                    st.error("No English text", icon="❌")
                    st.markdown("Some introductory English text can help inform the viewer and immediately hook them in.")
                    st.error("No strong emotional features", icon="❌")
                    st.markdown("A strong emotional hook, usually in the form of an exclaiming or frowning face, can significantly improve your chances with the algorithm and quickly steal the viewers' attention.")
        elif perc<10:
            st.balloons()
            st.subheader(f"Damn... You scored {perc}% !")
            st.markdown("<h4>Your thumbnail is likely to attract <b>no</b> viewership.</h4>", unsafe_allow_html=True)
            imgfin1,imgfin2,imgfin3 = st.columns(3)
            with imgfin2:
                st.image("https://flipanim.com/gif/m/k/mkFPUszT.gif")
            right,wrong = st.columns(2)
            with right:
                with st.expander("**THINGS YOU GOT RIGHT**"):
                    st.success("You tried!", icon="🔥")
            with wrong:
                with st.expander("**THINGS TO IMPROVE**"):
                    st.error("Interesting choice for a thumbnail!", icon="‼")
                    st.markdown("Maybe try improving your colour composition, imbue some more emotion, maybe sprinkle in some text and try again!")
except:
    st.title("You're not logged in")
    st.subheader("Please navigate to the login page from the sidebar to log in.")
    try:
        f.close()
        os.remove("logged_in.json")
    except:
        pass