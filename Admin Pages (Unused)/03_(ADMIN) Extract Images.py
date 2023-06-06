# IMPORTING MODULES
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import urllib.request

import urllib.parse as p
import re
import os
import pickle

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

import streamlit as st
from PIL import Image

st.title("Extracting Data From YouTube API!")


# The following function is used to authenticate with the google cloud and gain permission and access
# to the Youtube DATA API
@st.cache_data
def youtube_authenticate():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service = "youtube"
    api_version = "v3"
    client_secrets_file = "YT_API_Credentials.json"
    creds = None
    
    # A file named token.pickle shall be automatically created in the folder which stores the user's access and refresh 
    # tokens created automatically when the authorization flow completes for the first time
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, allow the user to log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
        # save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build(api_service, api_version, credentials=creds)

# Authenticate with YouTube API
authenticate_state = st.text("Authenticating...")
youtube = youtube_authenticate()
authenticate_state.text("Authenticating... Done!")

func_state = st.text("Defining some useful functions...")
# The following function will help us extract the youtube video ID from the video URL

@st.cache_data
def get_video_id_by_url(url):
    """
    Return the Video ID from the video `url`
    """
    # split URL parts
    parsed_url = p.urlparse(url)
    # get the video ID by parsing the query of the URL
    video_id = p.parse_qs(parsed_url.query).get("v")
    if video_id:
        return video_id[0]
    else:
        raise Exception(f"Unable to parse video URL: {url}")
        
# The below function recieves a YouTube service object (returned from youtube_authenticate() function) 
# and a keyword argument accepted by the API. It then returns the API response for a specific video
@st.cache_data
def get_video_details(_youtube, **kwargs):
    return youtube.videos().list(
        part="snippet,contentDetails,statistics",
        **kwargs
    ).execute()

# A function that takes a response returned from the above get_video_details() function
# and returns the video title
@st.cache_data
def get_video_title(video_response):
    items = video_response.get("items")[0]
    snippet = items["snippet"]
    #title         = snippet["title"]

    # Replace '|' with '-' to prevent errors in saving files
    #title         = snippet["title"].replace("|","-")
    title = [idx for idx in snippet["title"].replace("|","-") if not re.findall("[^\u0000-\u05C0\u2100-\u214F]+", idx)]
    title1 = ''.join(title).replace("/", "")
    title1 = title1.replace("?", "")
    title1 = title1.replace('"', '')
    title1 = title1.replace(':', '')
    return title1

# A function that takes a response returned from the above get_video_details() function
# and returns the channel title
@st.cache_data
def get_channel_title(video_response):
    items = video_response.get("items")[0]
    snippet = items["snippet"]
    
    channel_title = snippet["channelTitle"]
    return channel_title

# A function that takes a response returned from the above get_video_details() function
# and returns the video description
@st.cache_data
def get_description(video_response):
    items = video_response.get("items")[0]
    snippet = items["snippet"]
    
    description   = snippet["description"]
    return description

# A function that takes a response returned from the above get_video_details() function
# and returns the number of comments on the video
@st.cache_data
def get_comment_count(video_response):
    items = video_response.get("items")[0]
    statistics = items["statistics"]
    
    comment_count = statistics["commentCount"]
    return comment_count

# A function that takes a response returned from the above get_video_details() function
# and returns the number of views the video has
@st.cache_data
def get_view_count(video_response):
    items = video_response.get("items")[0]
    statistics = items["statistics"]
    
    view_count = statistics["viewCount"]
    return view_count

# A function that takes a response returned from the above get_video_details() function
# and returns the duration of the video
@st.cache_data
def get_duration(video_response):
    items = video_response.get("items")[0]
    content_details = items["contentDetails"]

    duration = content_details["duration"]
    # Duration provided by API has a format similar to - 'PT5H50M15S'
    
    # Parsing it below to convert it into a format similar to - '5:50:15'
    parsed_duration = re.search(f"PT(\d+H)?(\d+M)?(\d+S)", duration).groups()
    duration_str = ""
    for d in parsed_duration:
        if d:
            duration_str += f"{d[:-1]}:"
    duration_str = duration_str.strip(":")
    
    return duration_str

func_state.text('Defining some functions... Done!')

#-------------------------------------------------------------------

# Code to scrape video urls from youtube trending page to be added here. 
# Depending on Google Vision API, either all trending videos shall be extracted together
# Or trending videos shall be extracted for each category seperately
# These scraped urls can be used using the code blocks below to get video details and thumbnails

#st.markdown("<h3>Enter below the URL of the YouTube Website you wish to retrieve details from: </h3>", unsafe_allow_html=True)
#st.markdown("<h6>For example: 'https://www.youtube.com/feed/trending'</h6>", unsafe_allow_html=True)

#st.text_input("URL: ", key='url')
web_page_url = 'https://www.youtube.com/feed/trending'

url_state = st.text('Extracting URLs of videos from the webpage.')

fid = urllib.request.urlopen(web_page_url)
url_state.text('Extracting URLs of videos from the webpage..')
webpage=fid.read().decode('utf-8')
url_state.text('Extracting URLs of videos from the webpage...')
# This shall contain all urls for trending videos
url_df = []

for line in webpage.split('"'):
    if '/watch?v' in line:
        if len(line) ==20:
            url_df.append("https://www.youtube.com"+line)
            
url_state.text('Extracting URLs of videos from the webpage... Done!')

#-------------------------------------------------------------------

# Code segment to download video thumbnail

import requests # Used to request image from the website
import shutil # Used to save image locally
import os

st.text("Making directories if they don't exist...")
if not os.path.exists("thumbnail_data"):
    os.makedirs("thumbnail_data")

# Set the https format for youtube thumbnails
thumb_address = "https://img.youtube.com/vi/"
thumb_format = "/0.jpg"

# Next step - get website links from url_df and loop the following code. Maybe make use of multi threading for downloads
jj=0
final_state = st.text("Now, extracting the thumbnails of each video: ")
latest_iteration = st.empty()
bar = st.progress(0)

col1, col2 = st.columns(2)

st.sidebar.markdown("<h2>Extracted Thumbnails Preview: ",unsafe_allow_html=True)

for i in url_df:
    video_url = i
    # parse video ID from URL
    video_id = get_video_id_by_url(video_url)
    # make API call to get video info
    response = get_video_details(youtube, id=video_id)

    thumbnail_url = thumb_address+video_id+thumb_format

    video_title = get_video_title(response)
    file_name = 'thumbnail_data/{}-{}.jpg'.format(jj, video_title)
    res = requests.get(thumbnail_url, stream = True)

    if res.status_code == 200:
        with open(file_name,'wb') as f:
            shutil.copyfileobj(res.raw, f)
        print('Image sucessfully Downloaded: ',file_name.format())
    else:
        print('Image could not be Downloaded')
        
    with st.sidebar:
        st.markdown(video_title)
        image = Image.open(file_name.format())
        st.image(image)
        st.markdown('<hr>',unsafe_allow_html=True)
    
    jj=jj+1
    jjprogress = jj/len(url_df)
    latest_iteration.text(f'Progress: {round(jjprogress*100,2)}%')
    bar.progress(jjprogress)
st.sidebar.write(f"{len(url_df)} thumbnails extracted.")
    
st.success("All done!", icon="✔️")