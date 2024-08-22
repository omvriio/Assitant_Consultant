import os
import streamlit as st
import google.generativeai as genai
import dotenv 
from time import sleep
# Load environment variables from .env file
dotenv.load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    layout="wide",
)

# Add custom CSS to reduce margin and padding and hide icons
st.markdown(
    """
    <style>
    [data-testid="Welcome-message"] {}
    [data-testid="tit"] {position: fixed;top: 0;left: 0;right: 0;z-index: 1000;background-color: #262730;padding: 10px;color: white; font-size:50px;}
    #MainMenu {background: #00000;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    [data-testid="stAppViewBlockContainer"]{padding: 0 1rem;}
    [data-testid="chatAvatarIcon-user"]{width:10px;}
    [data-testid="chatAvatarIcon-assistant"]{width:10px;}
    [data-testid="stchatMessage"].st-emotion-cache-4oy321 eeusbqq4{padding:0;}
    [id="welcome-to-capge-ai"] {height: 10px;}
    [data-testid="stAppViewBlockContainer"]{padding:0 5px 0 0;}
    h1#welcome {padding: 0 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Conditionally render the container
   

def clear_chat():
    st.session_state.chat_session.history.clear()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return ("AI", "UI/agent.png")
    else:
        return (user_role, "UI/person.png")

# Display the chat history
for message in st.session_state.chat_session.history:
    x, y = translate_role_for_streamlit(message.role)
    with st.chat_message(name=x, avatar=y):
        st.markdown(message.parts[0].text)
with st._bottom.container():
    # Input field for user's message
    st.button(":material/clear_all: clear", on_click=clear_chat)
    user_prompt = st.chat_input("Ask me...")
t = st.container()

#create a button at the bottom of the page
if user_prompt:
    # Update the flag when the first user input is received
    # Add user's message to chat and display it
    st.chat_message(name="human", avatar="UI/person.png").markdown(user_prompt)
    with st.spinner("Thinking ..."):
    # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

    # Display Gemini-Pro's response
    with st.chat_message("AI", avatar="UI/agent.png"):
        st.markdown(gemini_response.text)
else:
    t.title(">> Welcome !")
    t.write("""> Welcome to CAPGE AI - Your Personal Assistant!
    Ask any development-related question here and receive detailed guidance to help you build, debug, and optimize your code efficiently.""")