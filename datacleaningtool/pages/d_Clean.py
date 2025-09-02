import sys
import os

# --- Ensure imports resolve relative to repo root ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from pathlib import Path
import traceback
import pandas as pd
import numpy as np
import base64
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass

# ==================================
# PAGE CONFIGURATION
# ==================================
st.set_page_config(page_title="Data Cleaner | ScrubHub", layout="wide", initial_sidebar_state="collapsed")

# --- Asset Directory ---
try:
    ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
except NameError:
    ASSETS_DIR = Path("assets")

# ==================================
# FALLBACKS (used if real deps fail)
# ==================================
@dataclass
class FallbackCleaningSummary:
    rows_affected: int = 0
    action_taken: str = "Fallback Action"

class FallbackDataCleaningBot:
    @staticmethod
    def respond(_prompt: str, df: pd.DataFrame, **_kwargs) -> Tuple[str, pd.DataFrame, Optional[Dict[str, Any]]]:
        return "AI is in fallback mode. Please check the application logs.", df.copy(), None

def fallback_init_session_state():
    st.warning("Critical error: Session state is using a fallback initializer.")
    keys_defaults = {
        'df': None,
        'original_df': None,
        'ai_suggestions': {},
        'cleaning_steps': [],
        'trash_data': {},
        'show_review_button': False,
        'ai_has_cleaned': False,
        'data_quality_score': None,
        'ai_chat_history': [],
        'cleaning_bot': FallbackDataCleaningBot(),
        'initialized': True
    }
    for key, default in keys_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def fallback_log_cleaning_step(action: str, details=None):
    print(f"FALLBACK_LOG: Action='{action}' Details={details if details is not None else {}}")

# ==================================
# IMPORT UTILITIES WITH ERROR HANDLING
# ==================================
try:
    from styles.d_clean_style import get_css
    from utils.cleaning_utils.ai_utils import DataCleaningBot
    from utils.cleaning_utils.manual_utils import (
        analyze_dataset,
        remove_duplicates,
        handle_missing_values,
        clean_data_types,
        standardize_values,
        normalize_dates,
        CleaningSummary
    )
    from utils.shared import init_session_state, log_cleaning_step
except ImportError as e:
    st.error(f"A critical import failed: {e}. The application will run in a limited fallback mode.")
    # Assign fallbacks if primary imports fail
    def get_css(_):
        return "<style>body {background-color: #0f172a; color: #e2e8f0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial}</style>"
    DataCleaningBot = FallbackDataCleaningBot
    init_session_state = fallback_init_session_state
    log_cleaning_step = fallback_log_cleaning_step

# ==================================
# STYLING APPLICATION
# ==================================
def apply_styling(background_image_path: Path):
    bg_css_property = "background-color: #0f172a;"
    if background_image_path.is_file():
        try:
            with open(background_image_path, "rb") as f:
                base64_bg = base64.b64encode(f.read()).decode()
            bg_css_property = (
                f'background-image: url("data:image/jpeg;base64,{base64_bg}"); '
                f'background-size: cover; background-attachment: fixed;'
            )
        except Exception as e:
            print(f"ERROR (d_Clean.py): Could not load background image: {e}")
    if callable(get_css):
        st.markdown(get_css(bg_css_property), unsafe_allow_html=True)

apply_styling(ASSETS_DIR / "background_blue.jpg")

# ==================================
# HELPERS
# ==================================
def _ensure_arrow_safe_dtypes(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Make DataFrame Arrow-safe for Streamlit (avoid object+NaN integer issues)."""
    if df is None:
        return None
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str)
        if pd.api.types.is_integer_dtype(df_copy[col]) and df_copy[col].isnull().any():
            df_copy[col] = df_copy[col].astype(float)
    return df_copy

def _update_trash_data(action_key: str, new_trash_df: Optional[pd.DataFrame]):
    """Store dropped rows for possible restore/export."""
    if 'trash_data' not in st.session_state or not isinstance(st.session_state.trash_data, dict):
        st.session_state.trash_data = {}
    if new_trash_df is not None and not new_trash_df.empty:
        safe_key = "".join(c if c.isalnum() else "_" for c in action_key).lower()
        st.session_state.trash_data[safe_key] = new_trash_df.copy()
        st.toast(f"Captured {len(new_trash_df)} removed rows from '{safe_key}'.")

def _apply_manual_cleaning_action(action_name_key: str, cleaning_function: Callable, df_input: pd.DataFrame,
                                  action_description: str, **kwargs: Any):
    """Run one manual tool and update state."""
    if df_input is None:
        st.warning("No data loaded to perform the action on.")
        return
    try:
        if not callable(cleaning_function):
            st.error(f"Cleaning tool '{action_description}' is currently unavailable.")
            return

        df_cleaned, report_obj, trash_from_step = cleaning_function(df_input.copy(), **kwargs)
        st.session_state.df = _ensure_arrow_safe_dtypes(df_cleaned)

        if isinstance(trash_from_step, pd.DataFrame):
            _update_trash_data(action_name_key, trash_from_step)

        # Enable the big review button
        st.session_state.show_review_button = True
        st.session_state.ai_has_cleaned = True

        st.success(f"Action '{action_description}' applied successfully.")
        log_cleaning_step(action=f"Manual: {action_description}", details=getattr(report_obj, '__dict__', {}))
        st.rerun()

    except Exception as e:
        st.error(f"Operation '{action_description}' failed: {e}")
        print(traceback.format_exc())

# ==================================
# CHAT + AI
# ==================================
def render_chat_interface() -> None:
    """Chat UI with AI; shows inline download + enables bottom Review button after cleaning."""
    # Render previous history
    for msg in st.session_state.ai_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Chat input for user
    prompt = st.chat_input("Ask Scrubbie to clean your data or just chat…")

    if not prompt:
        return

    # Append user message to history and show in UI
    st.session_state.ai_chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message container
    with st.chat_message("assistant"):
        placeholder = st.empty()

        def stream_to_ui(text: str):
            """Stream partial response into UI."""
            placeholder.markdown(text + "▌", unsafe_allow_html=True)

        try:
            # Prepare the AI bot
            bot = st.session_state.cleaning_bot
            df_for_bot = st.session_state.df.copy() if st.session_state.df is not None else pd.DataFrame()
            user_info = st.session_state.get("user_info", {})
            user_id = user_info.get("id") if user_info else None

            # Call AI respond() with streaming
            final_explanation, final_df, ai_action_result = bot.respond(
                user_input=prompt,
                df=df_for_bot,
                ui_update_callback=stream_to_ui
            )

            # Display final AI response
            placeholder.markdown(final_explanation, unsafe_allow_html=True)

            # Store assistant message in history
            st.session_state.ai_chat_history.append({"role": "assistant", "content": final_explanation})

            # Update DataFrame if AI returned one
            if isinstance(final_df, pd.DataFrame):
                st.session_state.df = _ensure_arrow_safe_dtypes(final_df)

            # Handle AI UI signals (from ai_action_result)
            if isinstance(ai_action_result, dict):
                # Enable review button if cleaning completed
                if ai_action_result.get("ui_action") == "show_review_button":
                    st.session_state.show_review_button = True
                    st.session_state.ai_has_cleaned = True

                    # Show success message
                    st.success("✅ Cleaning complete!")

                    # Offer download of the cleaned DataFrame
                    if isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty:
                        try:
                            cleaned_csv = st.session_state.df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="⬇ Download Cleaned Data",
                                data=cleaned_csv,
                                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.info(f"(Could not generate CSV for download: {e})")

                # Handle prefilled tool suggestions
                if ai_action_result.get("ui_action") == "prefill_tool":
                    st.session_state.ai_suggestions = ai_action_result.get("details", {})
                    st.toast("Scrubbie pre-configured a tool in the 'Manual Cleaning Tools' tab!")

                # Handle any trash frames returned by AI
                for key, value in ai_action_result.items():
                    if key.startswith("trash_") and isinstance(value, pd.DataFrame) and not value.empty:
                        _update_trash_data(key, value)

            # Log this interaction
            log_cleaning_step(action="AI Chat", details={"prompt": prompt, "response": final_explanation})

        except Exception as e:
            placeholder.error("An error occurred while communicating with the AI.")
            print(f"ERROR in chat interface: {e}\n{traceback.format_exc()}")

# ==================================
# MANUAL CLEANING UI
# ==================================
def render_manual_cleaning_interface() -> None:
    st.header("Manual Cleaning Tools")

    if st.session_state.df is None:
        st.warning("No data loaded.")
        return

    # Helper for select labels: show missing count
    def format_column_with_missing_count(col_name):
        missing_count = st.session_state.df[col_name].isnull().sum()
        return f"{col_name} ({missing_count} missing)" if missing_count > 0 else col_name

    tab_dup, tab_miss, tab_std, tab_type, tab_rename, tab_dates = st.tabs(
        ["Duplicates", "Missing Values", "Text Standardization", "Data Types", "Rename Columns", "Date Normalization"]
    )

    # --- Duplicates ---
    with tab_dup:
        st.subheader("Manage Duplicate Rows")

        col1, col2 = st.columns([2, 1])
        with col1:
            total_duplicates = st.session_state.df.duplicated().sum()
            st.metric("Duplicates Found", total_duplicates)

            opts = {"Keep First": "first", "Keep Last": "last", "Remove All Occurrences": False}
            selected_strategy = st.selectbox("Strategy:", opts.keys(), key="dup_strategy",
                                             help="Choose which duplicate row to keep.")
        with col2:
            st.write("")
            st.write("")
            if st.button("Remove Duplicates", key="remove_dup_btn", use_container_width=True, type="primary"):
                _apply_manual_cleaning_action("remove_duplicates", remove_duplicates, st.session_state.df,
                                              "Remove Duplicates", keep=opts[selected_strategy])

    # --- Missing Values ---
    with tab_miss:
        st.subheader("Handle Missing Values")

        all_columns = st.session_state.df.columns.tolist()
        columns_with_missing = [c for c in all_columns if st.session_state.df[c].isnull().sum() > 0]

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Select Columns**")
            if not columns_with_missing:
                st.info("No missing values found in any column.")
                selected_cols = []
            else:
                selected_cols = st.multiselect(
                    "Target columns with missing data:",
                    options=columns_with_missing,
                    format_func=format_column_with_missing_count,
                    help="Only columns that currently contain missing values are shown here."
                )

        with col2:
            st.write("**Choose Action**")
            opts = {"Fill with Mode": "mode", "Fill with Median": "median", "Fill with Custom Value": "custom",
                    "Drop Rows": "drop_row_any"}
            selected_display = st.selectbox("Strategy:", opts.keys(), key="miss_strategy")
            selected_key = opts[selected_display]

            custom_val = ""
            if selected_key == "custom":
                custom_val = st.text_input("Custom Fill Value:", key="miss_custom_val")

            if st.button("Process Missing Values", key="process_miss_btn", use_container_width=True, type="primary",
                         disabled=not columns_with_missing):
                if not selected_cols:
                    st.warning("Please select at least one column to process.")
                elif selected_key == "custom" and not str(custom_val).strip():
                    st.warning("Please provide a custom fill value.")
                else:
                    _apply_manual_cleaning_action("handle_missing_values", handle_missing_values, st.session_state.df,
                                                  "Handle Missing Values", strategy=selected_key,
                                                  column_to_process=selected_cols,
                                                  fill_value=custom_val if selected_key == "custom" else None)

    # --- Text Standardization ---
    with tab_std:
        st.subheader("Standardize Text Columns")
        col1, col2 = st.columns(2)

        with col1:
            text_cols = list(st.session_state.df.select_dtypes(include=['object', 'string']).columns)
            target_cols = st.multiselect("Columns to standardize:", options=text_cols, default=text_cols)

        with col2:
            st.write("**Standardization Options**")
            strip_ws = st.toggle("Strip leading/trailing whitespace", value=True)
            to_lower = st.toggle("Convert to lowercase")
            to_upper = st.toggle("Convert to UPPERCASE", disabled=to_lower)
            remove_punct = st.toggle("Remove punctuation ([.,!?;:])")
            remove_digits = st.toggle("Remove digits (0-9)")

        if st.button("Standardize Text", key="standardize_text_btn", use_container_width=True, type="primary"):
            if not target_cols:
                st.warning("Please select columns to standardize.")
            else:
                chars_to_remove = ""
                if remove_punct:
                    chars_to_remove += r'[.,!?;:]'
                if remove_digits:
                    chars_to_remove += r'\d'
                regex_to_remove = f"[{chars_to_remove}]" if chars_to_remove else None

                _apply_manual_cleaning_action("standardize_values", standardize_values, st.session_state.df,
                                              "Standardize Text",
                                              target_columns=target_cols,
                                              text_to_lowercase=to_lower,
                                              text_to_uppercase=to_upper,
                                              strip_whitespace_all=strip_ws,
                                              remove_chars_regex=regex_to_remove)

    # --- Date Normalization ---
    with tab_dates:
        st.subheader("Normalize Date Formats")

        df = st.session_state.df
        likely_date_cols = [col for col in df.columns
                            if (df[col].dtype == 'object') or pd.api.types.is_datetime64_any_dtype(df[col].dtype)]

        if not likely_date_cols:
            st.info("No columns suitable for date normalization were found.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                default_dates = [c for c in likely_date_cols
                                 if pd.api.types.is_datetime64_any_dtype(df[c].dtype) or 'date' in c.lower()]
                selected_cols = st.multiselect("Columns to normalize:",
                                               options=likely_date_cols,
                                               default=default_dates,
                                               help="Select the columns you want to convert to a standard date format.")
            with col2:
                target_format = st.text_input("Target Format:", value="%Y-%m-%d",
                                              help="Use strftime codes. Common: %Y-%m-%d or %d-%m-%Y.")

            if st.button("Normalize Selected Dates", key="normalize_dates_btn", use_container_width=True, type="primary"):
                if not selected_cols:
                    st.warning("Please select at least one column to normalize.")
                elif not target_format:
                    st.warning("Please provide a target format.")
                else:
                    _apply_manual_cleaning_action(
                        "normalize_dates",
                        normalize_dates,
                        st.session_state.df,
                        "Normalize Dates",
                        columns_to_process=selected_cols,
                        target_format=target_format
                    )

    # --- Data Types ---
    with tab_type:
        st.subheader("Optimize Data Types")
        st.info("Attempts to convert columns to more efficient types (numbers/dates) for memory and performance.")

        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("Attempt to Optimize All Data Types", key="optimize_types_btn", use_container_width=True,
                         type="primary"):
                _apply_manual_cleaning_action("clean_data_types", clean_data_types, st.session_state.df,
                                              "Optimize Data Types")

    # --- Rename Columns ---
    with tab_rename:
        st.subheader("Rename Columns")
        st.write("Edit the column names below and click 'Apply Changes' to update the dataset.")

        col_names = st.session_state.df.columns.tolist()
        new_names = {}

        grid_cols = st.columns(3)
        for i, col_name in enumerate(col_names):
            with grid_cols[i % 3]:
                new_names[col_name] = st.text_input(f"**{col_name}**", value=col_name, key=f"rename_{col_name}")

        if st.button("Apply Renaming", key="rename_cols_btn", use_container_width=True, type="primary"):
            rename_map = {old: new for old, new in new_names.items() if old != new}
            if not rename_map:
                st.warning("No column names were changed.")
            else:
                try:
                    original_df_copy = st.session_state.df.copy()
                    st.session_state.df.rename(columns=rename_map, inplace=True)
                    log_cleaning_step(action="Manual: Rename Columns", details=rename_map)
                    st.success("Columns renamed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to rename columns: {e}")
                    st.session_state.df = original_df_copy  # Restore on failure

# ==================================
# MAIN PAGE LAYOUT
# ==================================
def main_page_layout() -> None:
    # --- Initialize session state once ---
    if 'initialized' not in st.session_state:
        try:
            init_session_state()
        except Exception as e:
            fallback_init_session_state()
            print(f"Error initializing session state from shared file: {e}")

    # Flags we rely on
    if 'show_review_button' not in st.session_state:
        st.session_state.show_review_button = False
    if 'ai_has_cleaned' not in st.session_state:
        st.session_state.ai_has_cleaned = False

    # --- Init the AI bot ---
    if 'cleaning_bot' not in st.session_state or st.session_state.cleaning_bot is None:
        try:
            st.session_state.cleaning_bot = DataCleaningBot()
        except Exception as e:
            st.session_state.cleaning_bot = FallbackDataCleaningBot()
            st.sidebar.error("AI Bot failed to initialize.")
            print(f"ERROR: Could not initialize DataCleaningBot: {e}")

    # --- Header / Logo ---
    logo_path = ASSETS_DIR / "logo_wit.png"
    if logo_path.is_file():
        try:
            with open(logo_path, "rb") as f:
                logo_base64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f'<div class="logo-container"><img src="data:image/png;base64,{logo_base64}" class="logo-image" alt="ScrubHub Logo"></div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            print(f"WARNING (d_Clean.py): Logo display error: {e}")

    st.markdown("<div class='page-header-container'><h1>ScrubHub AI & Manual Data Cleaner</h1></div>",
                unsafe_allow_html=True)

    # --- Guard when no data loaded ---
    if 'df' not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
        st.info("Welcome! Please upload your data from the 'Input' page to begin cleaning.")
        if st.button("Go to Data Input Page"):
            st.switch_page("pages/c_Input.py")
        st.stop()

    # --- Main tabs ---
    tab1, tab2 = st.tabs(["AI Cleaning Assistant", "Manual Cleaning Tools"])
    with tab1:
        render_chat_interface()
    with tab2:
        render_manual_cleaning_interface()

    # --- Bottom: Proceed to Review ---
    if st.session_state.get('show_review_button', False):
        st.divider()
        st.markdown("<h3 style='text-align: center;'>Proceed with Cleaned Data</h3>", unsafe_allow_html=True)
        _, review_col, _ = st.columns([2, 3, 2])
        with review_col:
            if st.button("Review Cleaned Data & Export Options", type="primary", use_container_width=True):
                # Reset flags so the user can continue chatting if they return
                st.session_state.show_review_button = False
                st.session_state.ai_has_cleaned = False
                st.switch_page("pages/e_Datareview.py")

# --- Script entrypoint ---
if __name__ == "__main__":
    main_page_layout()
