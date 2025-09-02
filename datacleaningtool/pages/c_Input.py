"""
ScrubHub - Data Ingestion Portal
Accepts files and facilitates database connections to load data for cleaning.
"""
import streamlit as st
import pandas as pd
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List
import traceback
import time

# Attempt to import UploadedFile directly for cleaner type hinting
try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
except ImportError:
    UploadedFile = Any # Fallback type

# Attempt to import shared utilities
try:
    from utils.shared import init_session_state
except ImportError:
    def init_session_state():
        st.warning("Fallback init_session_state used in c_Input.py. Ensure utils.shared is correctly configured.")
        # Full fallback init_session_state
        for key, default_val in [('df', None), ('original_df', None), ('ai_suggestions', {}), ('cleaning_steps', []), ('trash_data', {}), ('cleaning_action_performed', False), ('data_quality_score', None)]:
            if key not in st.session_state: st.session_state[key] = default_val
        if 'ai_chat_history' not in st.session_state: st.session_state.ai_chat_history = []
        if 'cleaning_bot' not in st.session_state: st.session_state.cleaning_bot = None
        if 'initialized' not in st.session_state: st.session_state.initialized = True

# Attempt to import database utilities
MYSQL_UTILS_AVAILABLE = False
try:
    from utils.mysql_utils import get_mysql_connection, fetch_data
    MYSQL_UTILS_AVAILABLE = True
except ImportError:
    print("WARNING (c_Input.py): mysql_utils not found. DB connection will be limited.")
    def get_mysql_connection(*_args, **_kwargs):
        st.error("Database connection utility (get_mysql_connection) not available.")
        raise NotImplementedError("MySQL utils (get_mysql_connection) not available.")
    def fetch_data(*_args, **_kwargs):
        st.error("Data fetching utility (fetch_data) not available.")
        raise NotImplementedError("MySQL utils (fetch_data) not available.")

# --- Asset Directory ---
try:
    ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
    if not ASSETS_DIR.is_dir():
        ASSETS_DIR = Path("assets")
except NameError:
    ASSETS_DIR = Path("assets")

# --- Page config (must come first) ---
st.set_page_config(page_title="Data Input | ScrubHub", layout="wide", initial_sidebar_state="collapsed")

# --- Set background image and page style ---
def set_page_style_with_background(image_filename: str):
    image_path = ASSETS_DIR / image_filename

    page_bg_css_property = "background-color: #001f3f;" # Donkerblauw fallback

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
        except Exception as e:
            print(f"ERROR (c_Input.py): Could not load background image '{image_filename}': {e}")
    else:
        print(f"WARNING (c_Input.py): Background image '{image_filename}' not found at {image_path}.")

    # --- Definitieve CSS met alle correcties ---
    st.markdown(f"""
        <style>
            .stApp {{
                {page_bg_css_property}
            }}
            .main .block-container {{
                background-color: rgba(0, 0, 0, 0.75) !important;
                padding: 2rem;
                border-radius: 12px;
            }}
            
            /* === ALGEMENE TEKSTKLEUR WIT MAKEN === */
            div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] h1,
            div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] h2,
            div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] h3,
            div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] h4,
            div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] p,
            p.page-subtitle,
            .stTabs [data-baseweb="tab"] button p,
            [data-testid="stFileUploader"] > label,
            .stAlert[data-testid="stInfo"] * {{
                color: white !important;
            }}
            /* "Database Configuration" (h6) specifiek wit maken */
            .main .block-container h6 {{
                color: white !important;
                margin-top: 1em;
                padding-bottom: 0.5em;
                border-bottom: 1px solid rgba(0, 123, 255, 0.3);
            }}

            /* --- UITZONDERINGEN (Waar tekst NIET wit moet zijn) --- */
            [data-testid="stFileUploadDropzone"] p {{
                color: #31333F !important; /* Donkere tekst in lichte dropzone */
            }}
            .stTextInput input, .stNumberInput input {{
                color: white !important; /* Witte tekst in donkere input velden */
                background-color: rgba(20, 30, 60, 0.7) !important;
                border: 1px solid #506090 !important;
            }}

            /* --- OVERIGE STYLING --- */
            [data-testid="stFileUploadDropzone"] {{
                background-color: #f0f2f6 !important;
                border: 2px dashed #cccccc !important;
            }}
            [data-testid="stFileUploadDropzone"] button {{
                background-color: white !important;
                color: #0056b3 !important;
                border: 1px solid #0056b3 !important;
            }}
            .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                background-color: rgba(0, 50, 100, 0.75) !important;
            }}
            
            /* Algemene knop stijl */
            .stButton>button {{ 
                color: white !important;
                background-color: #0056b3 !important; /* Sterk blauw */
                border: 1px solid #007bff !important;
                border-radius: 8px !important;
                padding: 0.6rem 1.2rem !important;
            }}
             .stButton>button:hover {{
                background-color: #007bff !important;
            }}
            
            /* Form Submit Knop ("Connect and Fetch Data") */
            [data-testid="stFormSubmitButton"] button {{
                background-color: #0056b3 !important;
                color: white !important;
                border: 1px solid #007bff !important;
                width: 100%;
            }}
            [data-testid="stFormSubmitButton"] button:hover {{
                background-color: #007bff !important;
                border-color: #50a0ff !important;
            }}
            
            [data-testid="stToast"] {{
                background-color: black !important; color: white !important;
            }}
        </style>
    """, unsafe_allow_html=True)

# --- Roep Styling Functie Aan ---
set_page_style_with_background("background_blue.jpg")

# --- Verberg Sidebar ---
st.markdown("""<style>[data-testid="stSidebar"] {display: none;}</style>""", unsafe_allow_html=True)

# --- Initialiseer Session State ---
try:
    if 'initialized' not in st.session_state:
        init_session_state()
except NameError:
    st.error("Critical error: `init_session_state` function not found.")
except Exception as e:
    st.error(f"Error during session state init: {e}")

# --- Functie om geladen DataFrame te verwerken ---
def process_loaded_data(new_df: pd.DataFrame, source_name: str):
    if not isinstance(new_df, pd.DataFrame) or new_df.empty:
        st.error(f"❌ Error: No data loaded from '{source_name}' or data is empty.")
        return

    with st.spinner("🔄 Finalizing data load and preparing session..."):
        st.session_state.df = new_df
        st.session_state.original_df = new_df.copy()
        st.session_state.cleaning_steps = []
        st.session_state.trash_data = {}
        st.session_state.ai_chat_history = [{"role": "assistant", "content": f"New dataset '{source_name}' loaded!"}]
        st.session_state.ai_suggestions = {}
        st.session_state.cleaning_action_performed = False
        st.session_state.data_quality_score = None

        st.toast(f"Dataset '{source_name}' (Shape: {new_df.shape}) ready!", icon="🎉")
        st.success(f"✅ Success! Loaded {len(new_df)} rows and {len(new_df.columns)} columns from '{source_name}'.")
        st.caption("Preview of the first 3 rows:")
        st.dataframe(new_df.head(3))

# --- Hoofd Pagina Inhoud ---
st.markdown("<h1 id='scrubhub-data-dropzone'>ScrubHub Data Dropzone </h1>", unsafe_allow_html=True)
st.markdown("<p class='page-subtitle'>Upload your files or connect to a database to begin the cleaning journey.</p>", unsafe_allow_html=True)
st.divider()

tab_upload, tab_db = st.tabs(["📤 Upload File", "🔌 Connect to Database"])

with tab_upload:
    st.subheader("Upload Your Data File")
    uploaded_file: Optional[UploadedFile] = st.file_uploader(
        "Drag & drop CSV or Excel files, or click to browse to begin.",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, Excel (.xlsx, .xls)",
        key="file_uploader_c_input_final_v2"
    )
    if uploaded_file is not None:
        try:
            with st.spinner(f"🔄 Reading '{uploaded_file.name}'..."):
                if uploaded_file.name.lower().endswith('.csv'): temp_df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')): temp_df = pd.read_excel(uploaded_file)
                else: st.error("Unsupported file type."); temp_df = None
            if temp_df is not None: process_loaded_data(temp_df, uploaded_file.name)
        except Exception as e: st.error(f"❌ Error processing file: {e}")

with tab_db:
    st.subheader("Connect to Database (MySQL Example)")
    if not MYSQL_UTILS_AVAILABLE:
        st.warning("Database connection functionality is currently unavailable.", icon="⚠️")
    else:
        with st.form(key="db_connection_form_cinput_final_v2"):
            st.markdown("<h6>⚙️ Database Configuration</h6>", unsafe_allow_html=True)
            db_host = st.text_input("Server Host", value=st.session_state.get("db_host", "localhost"))
            db_user = st.text_input("Username", value=st.session_state.get("db_user", "root"))
            db_password = st.text_input("Password", type="password")
            db_name = st.text_input("Database Name", value=st.session_state.get("db_name", "data_cleaner"))
            db_table_or_query = st.text_input("Table Name or SQL Query", value=st.session_state.get("db_table_or_query", "SELECT * FROM your_table"))

            submit_button = st.form_submit_button(label="🔗 Connect and Fetch Data")
        if submit_button:
            try:
                with st.spinner("🔄 Connecting to database..."):
                    is_query = "select" in db_table_or_query.lower() and "from" in db_table_or_query.lower()
                    conn = get_mysql_connection(db_host, db_user, db_password, db_name)
                    if conn:
                        db_df = fetch_data(conn, db_table_or_query, is_query=is_query)
                        conn.close()
                        if db_df is not None:
                            process_loaded_data(db_df, f"DB: {db_name}")
                        else:
                            st.warning("⚠️ No data returned from database.")
                    else:
                        st.error("❌ Failed to connect to database.")
            except Exception as e:
                st.error(f"❌ Database operation failed: {e}")
                print(traceback.format_exc())

# --- Navigatie knoppen ---
if 'df' in st.session_state and isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty:
    st.divider()
    st.markdown("<h4 style='text-align:center;'>Dataset Loaded Successfully!</h4>", unsafe_allow_html=True)
    nav_cols = st.columns([1, 2, 1])
    with nav_cols[1]:
        st.page_link("pages/d_Clean.py", label=" Proceed to Data Cleaning →", use_container_width=True)
else:
    st.divider()
    st.info("Upload a file or connect to a database to load your data. Options to proceed will appear once data is loaded.")