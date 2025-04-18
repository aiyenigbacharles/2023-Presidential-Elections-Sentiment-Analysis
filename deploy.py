import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import base64

# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Make text inputs and text areas have transparent backgrounds with just a border */
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {{
            background-color: rgba(0, 0, 0, 0.5) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
        }}
        /* Style the button */
        .stButton > button {{
            background-color: rgba(0, 0, 0, 0.6) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
        }}
        /* Remove the whitish background from main content area */
        .block-container {{
            background-color: transparent !important;
        }}
        /* Style the title and text to be more visible */
        h1, h2, h3, p, label, .stSelectbox label {{
            color: white !important;
            text-shadow: 1px 1px 2px black;
        }}
        /* Style the dropdown */
        .stSelectbox > div > div > div {{
            background-color: rgba(0, 0, 0, 0.5) !important;
            color: white !important;
        }}
        /* Remove any other containers that might have white backgrounds */
        .css-1d391kg, .css-12oz5g7, .css-1r6slb0, .css-keje6w, .css-1e5imcs {{
            background-color: transparent !important;
        }}
        /* Style the sidebar */
        .css-1oe6o3n, .css-1adrfps {{
            background-color: rgba(0, 0, 0, 0.7) !important;
        }}
        /* Style radio buttons */
        .stRadio > div {{
            background-color: transparent !important;
        }}
        .stRadio label {{
            color: white !important;
            text-shadow: 1px 1px 2px black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Function to add footer
def add_footer():
    footer = """
    <div class="footer">
        Developed by Charles Aiyenigba
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

# Load the model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('./updated_svm_model.joblib')
    
    # Load the vectorizer
    try:
        vectorizer = joblib.load('./updated_vectorizer.joblib')
    except FileNotFoundError:
        
        vectorizer = TfidfVectorizer(max_features=3000)  
    
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Function to predict sentiment
def predict_sentiment(text):
    # To preprocess the text using the same vectorizer used during training
    text_vectorized = vectorizer.transform([text])

    # To ensure the number of features matches what the model expects
    if text_vectorized.shape[1] != 3000:
        st.error(f"Error: Vectorizer produced {text_vectorized.shape[1]} features, but model expects 3000.")
        return None
    
    return model.predict(text_vectorized)[0]

# Main page
def main_page():
    st.title("2023 Presidential Elections Sentiment Analysis")
    
    text = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze"):
        sentiment = predict_sentiment(text)
        st.markdown(f"<h3>The Sentiment of this Text is: <span style='color:#ffdd00;'>{sentiment}</span></h3>", unsafe_allow_html=True)

# Visualization page
def visualization_page():
    st.title("Sentiment Analysis Visualizations")
    
    # Dropdown for selecting visualizations
    visualization = st.selectbox(
        "Select a visualization",
        ["Sentiment Distribution (Bar Chart)", "Sentiment Distribution (Pie Chart)", "Word Cloud", "Sentiment Count of Tweets (Atiku)", "Sentiment Count of Tweets (Obi)", "Sentiment Count of Tweets (Tinubu)",
         "Atiku Sentiment Count", "Obi Sentiment Count", "Tinubu Sentiment Count"]
    )
    
    if visualization == "Sentiment Distribution (Bar Chart)":
        # Placeholder for sentiment distribution chart
        st.image("./images/sentiment distribution of tweets - bar.png")
    elif visualization == "Sentiment Distribution (Pie Chart)":
        # Placeholder for word cloud image
        st.image("./images/sentiment distribution of tweets - pie.png")
    elif visualization == "Word Cloud":
        # Placeholder for word cloud image
        st.image("./images/word cloud.png")
    elif visualization == "Sentiment Count of Tweets (Atiku)":
        # Placeholder for atikus sentiment count
        st.image("./images/sentiment count - atiku.png")
    elif visualization == "Sentiment Count of Tweets (Obi)":
        # Placeholder for obis sentiment count
        st.image("./images/sentiment count - obi.png")
    elif visualization == "Sentiment Count of Tweets (Tinubu)":
        # Placeholder for tinubu sentiment count
        st.image("./images/sentiment count - tinubu.png")
    elif visualization == "Atiku Sentiment Count":
        # Placeholder for atiku sentiment count
        st.image("./images/sentiment count - atiku.png")
    elif visualization == "Obi Sentiment Count":
        # Placeholder for obi sentiment count
        st.image("./images/sentiment count - obi.png")
    elif visualization == "Tinubu Sentiment Count":
        # Placeholder for tinubu sentiment count
        st.image("./images/sentiment count - tinubu.png")

# Sidebar navigation
def sidebar():
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Sentiment Analysis", "Visualizations"])
    
    if page == "Sentiment Analysis":
        main_page()
    elif page == "Visualizations":
        visualization_page()

# Run the app
if __name__ == "__main__":
    # To add the background image
    add_bg_from_local('./images/flag.jpeg')
    sidebar()
    # To add the footer
    add_footer()