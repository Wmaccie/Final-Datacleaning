import streamlit as st
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import traceback


def init_session_state() -> None:
    """Initializes all required session state variables with default values."""
    session_keys = [
        ('df', None),
        ('original_df', None),
        ('cleaning_steps', []),
        ('cleaning_action_performed', False),
        ('trash_data', {}),
        ('ai_chat_history', []),
        ('cleaning_bot', None),
        ('ai_suggestions', {}),
        ('data_quality_score', None),
        ('user_info', None),
        ('show_review_button', False)
    ]
    for key, default_value in session_keys:
        if key not in st.session_state:
            st.session_state[key] = default_value
    st.session_state.initialized = True
    print("INFO (shared.py): Session state initialized.")


def log_cleaning_step(action: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Adds a cleaning action to the history log."""
    try:
        if 'cleaning_steps' not in st.session_state: st.session_state.cleaning_steps = []

        # <-- FIX: Robust handling for when user_info is None, preventing the 'NoneType' crash.
        user_info = st.session_state.get('user_info')
        user_email = (user_info or {}).get('email', 'anonymous')

        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action.strip(),
            'details': details or {},
            'user': user_email
        }
        if 'df' in st.session_state and isinstance(st.session_state.df, pd.DataFrame):
            entry['df_shape_after_action'] = st.session_state.df.shape
        st.session_state.cleaning_steps.append(entry)
    except Exception as e:
        print(f"ERROR (shared.py): Failed to log cleaning step '{action}': {e}\n{traceback.format_exc()}")


# --- De rest van het bestand blijft ongewijzigd ---

def get_cleaning_history(max_entries: int = 10) -> List[Dict[str, Any]]:
    try:
        history = st.session_state.get('cleaning_steps', [])
        if not isinstance(history, list):
            print("WARNING (shared.py): 'cleaning_steps' in session state is not a list. Returning empty history.")
            return []
        return [entry.copy() for entry in history[-max_entries:]]
    except Exception as e:
        print(f"ERROR (shared.py): Failed to retrieve cleaning history: {e}")
        return []


def validate_dataframe(df: Any) -> bool:
    if not isinstance(df, pd.DataFrame):
        print(f"VALIDATION_ERROR: Input is not a pandas DataFrame. Type: {type(df)}")
        return False
    if df.empty:
        print("VALIDATION_INFO: DataFrame is empty.")
        return False
    return True


def display_error(error: Exception, context: str = "", show_traceback: bool = True) -> None:
    try:
        st.error(f"**Error in {context or 'application'}:** {str(error)}")
        if show_traceback:
            with st.expander("Technical Details (Traceback)"):
                st.code(traceback.format_exc())
    except Exception as e_display:
        print(f"FALLBACK_DISPLAY_ERROR (Context: {context}): {str(error)}")
        print(f"Error during display_error itself: {e_display}")


if 'initialized' not in st.session_state:
    print("INFO (shared.py): 'initialized' flag not found. Calling init_session_state().")
    init_session_state()