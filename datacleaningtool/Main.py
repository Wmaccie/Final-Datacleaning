"""
I silently route users to our grand welcome page
"""
import streamlit as st

# Minimal configuration
st.set_page_config(
    page_title="ScrubHub",
    layout="wide",
    initial_sidebar_state="collapsed"  # Clean slate
)

# Redirect to the login page
st.switch_page("pages/a_LogIn.py")

