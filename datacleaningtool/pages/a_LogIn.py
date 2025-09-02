import streamlit as st
import base64
from pathlib import Path
import time
from typing import Optional, Dict, Any
import traceback  # For detailed error logging

# --- Asset Directory ---
try:
    ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
    if not ASSETS_DIR.is_dir():
        ASSETS_DIR = Path("assets")
except NameError:
    ASSETS_DIR = Path("assets")

# --- Attempt to import user management functions ---
MANAGER_UTILS_AVAILABLE = False
try:
    from utils.manager_utils import create_user, authenticate_user

    MANAGER_UTILS_AVAILABLE = True
except ImportError:
    print("WARNING (a_Login.py): manager_utils not found. Authentication/Signup will be disabled or use fallbacks.")


    def create_user(email: str, password_plain: str, is_admin: bool = False) -> bool:
        st.error("User creation utility (manager_utils.create_user) not available.")
        return False


    def authenticate_user(email: str, password_plain: str) -> Optional[Dict[str, Any]]:
        st.error("Authentication utility (manager_utils.authenticate_user) not available.")
        return None

# --- Page config (must come first) ---
st.set_page_config(page_title="Login | Signup - ScrubHub", layout="centered", initial_sidebar_state="collapsed")


# --- Set background image and page style ---
def set_page_style_with_background(image_filename: str):
    """Sets the background image for the entire app and adds custom styles for the login page."""
    image_path = ASSETS_DIR / image_filename

    page_bg_css_property = "background-color: #FFFFFF;"  # Default fallback

    if image_path.is_file():
        try:
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()
            img_ext = image_path.suffix.lower().replace('.', '')
            if img_ext == "jpg": img_ext = "jpeg"

            page_bg_css_property = f"""
                background-image: url("data:image/{img_ext};base64,{encoded_string}");
                background-size: cover;
                background-position: center center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            """
            print(f"INFO (a_Login.py): Background image '{image_filename}' loaded successfully.")
        except Exception as e:
            st.error(f"Could not load background image: {e}")
            print(f"ERROR (a_Login.py): Could not load background: {e}")
    else:
        st.warning(f"Background image '{image_filename}' not found at {image_path}. Using default background color.")
        print(f"WARNING (a_Login.py): Background image '{image_filename}' not found at {image_path}.")

    st.markdown(f"""
        <style>
            /* --- FORCE WHITE TEXT COLOR FOR THE ENTIRE PAGE --- */
            .stApp {{
                {page_bg_css_property}
            }}
            /* This is the main fix: Target all text elements on the page */
            h1, h2, h3, p, label, .stMarkdown, [data-testid="stText"] {{
                color: white !important;
            }}
            .main .block-container {{
                background-color: rgba(0, 0, 0, 0.75) !important;
                padding: 2.5rem 3rem;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            }}
            /* --- KEEP INPUT TEXT DARK FOR READABILITY --- */
            .stTextInput input, .stTextInput input[type="password"] {{
                color: #212121 !important; /* This text should remain dark */
                background-color: #FFFFFF !important;
                border: 1px solid #B0BEC5 !important;
                border-radius: 6px !important;
            }}
            .stTextInput input:focus, .stTextInput input[type="password"]:focus {{
                 border-color: #B71C1C !important;
                 box-shadow: 0 0 0 0.15rem rgba(183,28,28,0.35) !important;
            }}
            /* --- OTHER STYLES --- */
            .stTabs [data-baseweb="tab-list"] {{
                border-bottom: 2px solid rgba(183, 28, 28, 0.7);
            }}
            .stTabs [data-baseweb="tab"] p {{ /* Target the text inside the tab */
                color: #B0BEC5 !important;
            }}
            .stTabs [data-baseweb="tab"][aria-selected="true"] p {{ /* Target text of the selected tab */
                color: white !important;
            }}
            .stTabs [data-baseweb="tab"] {{
                background-color: rgba(255,255,255,0.1) !important;
                border-radius: 6px 6px 0 0 !important;
                margin-right: 4px;
            }}
            .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                background-color: rgba(183, 28, 28, 0.15) !important;
                border-bottom-color: #B71C1C !important;
            }}

            /* --- THIS IS THE FIX FOR THE LOGIN/CREATE BUTTONS --- */
            [data-testid="stFormSubmitButton"] button {{
                color: white !important; 
                background-color: #B71C1C !important; /* RODE ACHTERGROND */
                border: 1px solid #D32F2F !important; 
                border-radius: 6px !important;
                padding: 10px 18px !important; 
                font-weight: bold !important;
                transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
                width: 100%;
            }}
            [data-testid="stFormSubmitButton"] button:hover {{
                background-color: #D32F2F !important;
                border-color: #FFCDD2 !important;
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(183,28,28,0.4);
            }}

            /* Keep alert text colors as they are for contrast */
            .stAlert[data-testid="stSuccess"] * {{ color: #1B5E20 !important; }}
            .stAlert[data-testid="stError"] * {{ color: #C62828 !important; }}
            .stAlert[data-testid="stWarning"] * {{ color: #B26A00 !important; }}

            .stAlert[data-testid="stSuccess"] {{ background-color: #E8F5E9; border: 1px solid #A5D6A7; border-radius: 6px; }}
            .stAlert[data-testid="stError"] {{ background-color: #FFEBEE; border: 1px solid #EF9A9A; border-radius: 6px; }}
            .stAlert[data-testid="stWarning"] {{ background-color: #FFF3CD; border: 1px solid #FFDDA1; border-radius: 6px; }}
        </style>
    """, unsafe_allow_html=True)


# This call now matches the corrected function name above
set_page_style_with_background("background_blue.jpg")

# --- Hide sidebar ---
st.markdown("""<style>[data-testid="stSidebar"] {display: none;}</style>""", unsafe_allow_html=True)

# --- Page Title ---
st.markdown("<h1 id='scrubhub-account-portal'>ScrubHub Account Portal</h1>", unsafe_allow_html=True)

# --- Session state for user information ---
if 'user_info' not in st.session_state:
    st.session_state.user_info: Optional[Dict[str, Any]] = None

# --- Redirect if already logged in ---
if st.session_state.user_info:
    if not st.session_state.get("_login_redirect_attempted_", False):
        st.session_state._login_redirect_attempted_ = True
        st.info("You are already logged in. Redirecting to Welcome Page...")
        time.sleep(0.5)
        try:
            st.switch_page("pages/b_Welcome.py")
            st.stop()
        except Exception as e_nav_welcome_login_page_v3:
            st.error(f"Redirection failed: {e_nav_welcome_login_page_v3}")
            print(f"ERROR (a_Login.py): Redirection to Welcome page failed: {e_nav_welcome_login_page_v3}")
elif '_login_redirect_attempted_' in st.session_state:
    del st.session_state['_login_redirect_attempted_']

# --- Login/Signup Tabs ---
tab_login, tab_signup = st.tabs(["🔑 Login to Your Account", "📝 Create New Account"])

with tab_login:
    with st.form("login_form_scrubhub_main_v3", clear_on_submit=False):
        st.markdown("<h2 style='text-align: center;'>User Login</h2>", unsafe_allow_html=True)
        login_email = st.text_input("Email Address", key="login_email_field_main_v3", placeholder="you@example.com",
                                    autocomplete="email")
        login_password = st.text_input("Password", type="password", key="login_password_field_main_v3",
                                       placeholder="Enter your password", autocomplete="current-password")
        login_submit_button = st.form_submit_button("Login Securely")

    if login_submit_button:
        if not MANAGER_UTILS_AVAILABLE:
            st.error("Authentication service is temporarily unavailable. Please try again later.", icon="⚙️")
        elif login_email and login_password:
            try:
                with st.spinner("Authenticating your credentials..."):
                    user_data = authenticate_user(login_email, login_password)
                if user_data and isinstance(user_data, dict):
                    st.session_state.user_info = user_data
                    st.success("✅ Login successful! Redirecting to your ScrubHub dashboard...")
                    st.toast(f"Welcome back, {user_data.get('email', 'User')}!", icon="🎉")
                    time.sleep(1.0)
                    st.switch_page("pages/b_Welcome.py")
                else:
                    st.error("❌ Invalid email or password. Please check your details and try again.", icon="❗")
            except Exception as e_auth_login_main_v3:
                st.error(f"An unexpected error occurred during login: Please try again later.", icon="🔥")
                print(f"Login Exception (a_Login.py): {e_auth_login_main_v3}\n{traceback.format_exc()}")
        else:
            st.warning("⚠️ Please enter both your email and password to login.", icon="✍️")

with tab_signup:
    with st.form("signup_form_scrubhub_main_v3", clear_on_submit=True):
        st.markdown("<h2 style='text-align: center;'>Create Your ScrubHub Account</h2>", unsafe_allow_html=True)
        signup_email = st.text_input("Email Address", key="signup_email_field_main_v3", placeholder="you@example.com",
                                     autocomplete="email")
        signup_password = st.text_input("Choose a Password", type="password", key="signup_password_field_main_v3",
                                        placeholder="Minimum 8 characters")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_field_main_v3",
                                         placeholder="Re-enter your password")
        signup_submit_button = st.form_submit_button("Create My Account")

    if signup_submit_button:
        if not MANAGER_UTILS_AVAILABLE:
            st.error("Account creation service is temporarily unavailable. Please try again later.", icon="⚙️")
        elif signup_email and signup_password and confirm_password:
            if len(signup_password) < 8:
                st.error("❌ Password must be at least 8 characters long for security.", icon="🔒")
            elif signup_password == confirm_password:
                try:
                    with st.spinner("Creating your ScrubHub account..."):
                        success_creating_user = create_user(signup_email, signup_password, is_admin=False)

                    if success_creating_user:
                        st.success("✅ Account created successfully!")
                        st.info("Attempting to log you in automatically...")
                        time.sleep(0.5)
                        with st.spinner("Finalizing setup..."):
                            user_data_after_signup = authenticate_user(signup_email, signup_password)
                        if user_data_after_signup and isinstance(user_data_after_signup, dict):
                            st.session_state.user_info = user_data_after_signup
                            st.toast("Welcome to ScrubHub! Your account is ready.", icon="👋")
                            time.sleep(1.0)
                            st.switch_page("pages/b_Welcome.py")
                        else:
                            st.warning(
                                "Account created, but auto-login failed. Please proceed to the Login tab to sign in manually.",
                                icon="🔑")
                except ValueError as ve_signup_main_v3:
                    st.error(f"❌ Could not create account: {ve_signup_main_v3}", icon="🚫")
                except Exception as e_create_user_login_main_v3:
                    st.error(f"❌ An unexpected error occurred during signup: Please try again.", icon="🔥")
                    print(f"Signup Exception (a_Login.py): {e_create_user_login_main_v3}\n{traceback.format_exc()}")
            else:
                st.error("❌ Passwords do not match. Please re-enter carefully.", icon="❗")
        else:
            st.warning("⚠️ Please fill all required fields to create your account.", icon="✍️")

st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.2); margin-top: 2em;'>", unsafe_allow_html=True)
st.markdown("<p class='footer-text'>ScrubHub &copy; 2025 | Data Cleaning, Simplified.</p>", unsafe_allow_html=True)