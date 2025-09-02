import streamlit as st
import base64

# --- Page Config ---
st.set_page_config(
    page_title="Welcome to ScrubHub",
    layout="centered",  # Changed from "wide" to "centered"
    initial_sidebar_state="collapsed"
)

# --- Hide Streamlit default styling ---
hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp {margin: 0; padding: 0; height: 100vh; overflow: hidden;}  /* Prevent scrolling */
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- Set background ---
def set_background(image_file):
    with open(image_file, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{base64_img}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .welcome-content {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;  /* Ensure content fills the screen without scrolling */
                text-align: center;
                padding: 0 20px;
            }}
            .logo {{
                width: 300px;
                margin-bottom: 20px;
            }}
            .main-text {{
                color: white;
                font-size: 5.5rem;
                font-weight: bold;
                margin-bottom: 10px;
                line-height: 1.2;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }}
            .sub-text {{
                color: white;
                font-size: 1.5rem;
                margin-bottom: 40px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }}
            .start-button button {{
                background-color: #FF4B4B;
                color: white;
                font-size: 1.2rem;
                padding: 0.8rem 2rem;
                border: none;
                border-radius: 12px;
                transition: 0.3s;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }}
            .start-button button:hover {{
                background-color: #e03e3e;
                transform: scale(1.05);
            }}
        </style>
    """, unsafe_allow_html=True)

# --- Apply background ---
set_background("assets/background_blue.jpg")

# --- Main Content Container ---
st.markdown("""
<div class="welcome-content">
    <img src="data:image/png;base64,{}" class="logo">
    <div class="main-text">LET DATA LEAD THE WAY</div>
    <div class="sub-text">with ScrubHub</div>
    <div class="start-button">
""".format(base64.b64encode(open("assets/logo_wit.png", "rb").read()).decode()), unsafe_allow_html=True)

# Centered button
if st.button("Let's get started"):
    st.switch_page("pages/c_Input.py")

# Close containers
st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)