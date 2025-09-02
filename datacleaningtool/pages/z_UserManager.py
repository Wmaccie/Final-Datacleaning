import streamlit as st
import pandas as pd
from pathlib import Path
import base64
import datetime  # For formatting timestamps
import time  # For UI pauses
import traceback  # For detailed error logging
from typing import Dict, Any, List, Optional

# --- Attempt to import user management functions ---
USER_MANAGER_AVAILABLE = False
try:
    # --- THIS IS THE FIX ---
    # We now import all the required functions, including the new simulation
    from utils.manager_utils import (
        get_all_users,
        reset_user_password,
        toggle_admin_status,
        delete_user,
        get_simulated_api_usage_logs # The new simulation function must be in this list
    )
    USER_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"WARNING (z_Usermanager.py): Could not import from manager_utils: {e}. Using fallbacks.")
    # Fallback functions are defined below
    pass

# --- Fallback Functions (if imports fail) ---
if not USER_MANAGER_AVAILABLE:
    def get_all_users() -> List[Dict[str, Any]]:
        print("FALLBACK: get_all_users called")
        return [
            {'id': 1, 'email': 'admin@example.com', 'is_admin': True, 'created_at': '2025-01-01'},
            {'id': 2, 'email': 'user@example.com', 'is_admin': False, 'created_at': '2025-01-02'}
        ]
    def reset_user_password(uid, pwd) -> bool: return False
    def toggle_admin_status(uid, status) -> bool: return False
    def delete_user(uid) -> bool: return False
    def get_simulated_api_usage_logs() -> pd.DataFrame:
        print("FALLBACK: get_simulated_api_usage_logs called")
        return pd.DataFrame({
            'timestamp': [pd.Timestamp.now()],
            'email': ['fallback@example.com'],
            'model_used': ['gpt-4o-mini'],
            'cost_usd': [0.001]
        })

# --- Page Specific Configuration ---
PAGE_TITLE = "Admin Panel | User Management | ScrubHub"
PAGE_ICON = "🛡️"

# --- Asset Directory ---
try:
    ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
except NameError:  # Fallback if __file__ is not defined (e.g. in some Streamlit sharing/cloud environments)
    ASSETS_DIR = Path("assets")


# --- Page Styling (Consistent Vibe) ---
def set_usermanager_page_styling(background_image_path: Path):
    """Sets the styling for the User Management page."""
    bg_image_css_val = "background-color: #121212;"  # Default dark fallback if image fails
    if background_image_path.is_file():
        try:
            with open(background_image_path, "rb") as f_bg_admin_z:
                base64_bg_img = base64.b64encode(f_bg_admin_z.read()).decode()
            bg_image_css_val = f"""
                background-image: url("data:image/jpeg;base64,{base64_bg_img}");
                background-size: cover; background-position: center;
                background-repeat: no-repeat; background-attachment: fixed;
            """
        except Exception as e_bg_admin_page_z:
            print(f"ERROR (z_Usermanager.py): Loading background image '{background_image_path}': {e_bg_admin_page_z}")
    else:
        print(
            f"WARNING (z_Usermanager.py): Background image not found: '{background_image_path}'. Using fallback color.")

    page_styling_css = f"""
        <style>
            .stApp {{ {bg_image_css_val} color: white !important; }}
            h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, [data-testid="stText"],
            .stMetric > label, .stMetric div[data-testid="stMetricValue"], .stMetric div[data-testid="stMetricDelta"] {{
                 color: white !important; 
            }}
            .main .block-container {{ /* Main content block */
                background-color: rgba(10, 0, 0, 0.75) !important; /* Darker, slightly transparent red-tinted overlay */
                padding: 2rem;
                border-radius: 12px;
                border: 1px solid rgba(255, 82, 82, 0.3); /* Subtle red border */
            }}
            .stButton>button {{
                color: white !important; background-color: #B71C1C !important; /* ScrubHub Red */
                border: 1px solid #D32F2F !important; border-radius: 6px !important;
                padding: 8px 16px !important; font-weight: 500 !important; /* Adjusted font weight */
                transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            .stButton>button:hover {{ 
                background-color: #D32F2F !important; 
                border-color: #FFCDD2 !important;
                transform: translateY(-1px); 
                box-shadow: 0 4px 8px rgba(183,28,28,0.4);
            }}
            .stButton>button:active {{ background-color: #A30000 !important; transform: translateY(0px); }}

            .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
                color: #212121 !important; 
                background-color: #F5F5F5 !important; /* Lighter background for inputs */
                border: 1px solid #757575 !important; 
                border-radius: 6px !important;
            }}
            .stTextInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {{
                border-color: #B71C1C !important; /* Red border on focus */
                box-shadow: 0 0 0 0.2rem rgba(183,28,28,0.25);
            }}
            .stTextInput > label, .stSelectbox > label {{ 
                color: #E0E0E0 !important; /* Slightly off-white for labels */
                margin-bottom: 0.3rem; display: block; font-weight: 500;
            }}
            .stDataFrame {{ border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,82,82,0.2);}}
            .stDataFrame table th {{ background-color: #4a0e0e !important; color: white !important; }}
            .stDataFrame table td {{ color: #CFD8DC !important; background-color: rgba(30,30,30,0.7) !important; }}
            .stAlert[data-testid="stSuccess"] {{ background-color: #2E7D32; border-radius: 6px; color: white;}}
            .stAlert[data-testid="stError"] {{ background-color: #B71C1C; border-radius: 6px; color: white;}}
            .stAlert[data-testid="stWarning"] {{ background-color: #E65100; border-radius: 6px; color: white; }}
        </style>
    """
    st.markdown(page_styling_css, unsafe_allow_html=True)


# --- Page Config (must be the first Streamlit command AFTER imports) ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
# Call styling after set_page_config
set_usermanager_page_styling(ASSETS_DIR / "background_blue.jpg")


# ==================================
# AUTHENTICATION AND AUTHORIZATION CHECK
# ==================================
def display_access_denied_content():
    st.error("🚫 Access Denied!")
    access_denied_image_path = ASSETS_DIR / "access_denied.png"  # Create this image
    if access_denied_image_path.is_file():
        st.image(str(access_denied_image_path), width=200)
    st.warning("You do not have permission to view this page. This area is for administrators only.")

    col1_denied_nav, col2_denied_nav, _ = st.columns([1, 1, 2])
    with col1_denied_nav:
        if st.button("⬅️ Go to Welcome Page", key="admin_denied_goto_welcome_btn", use_container_width=True):
            st.switch_page("pages/b_Welcome.py")
    with col2_denied_nav:
        if st.button("🔑 Go to Login Page", key="admin_denied_goto_login_btn", use_container_width=True):
            if 'user_info' in st.session_state:  # Log out current non-admin user
                st.session_state.user_info = None
            st.switch_page("a_Login.py")
    st.stop()


# Perform the check immediately
if 'user_info' not in st.session_state or st.session_state.user_info is None:
    st.warning("🔒 You must be logged in to access this application. Redirecting to login...")
    time.sleep(1.5)
    st.switch_page("a_Login.py")
    st.stop()

current_user_info_data: Optional[Dict[str, Any]] = st.session_state.user_info
is_authorized_admin_flag = False

if isinstance(current_user_info_data, dict):
    if current_user_info_data.get('is_admin') is True:
        is_authorized_admin_flag = True
    elif current_user_info_data.get('email') == 'whitney@datacleaner.com':  # Fallback as requested
        is_authorized_admin_flag = True
        if not current_user_info_data.get('is_admin'):  # Log if DB flag is not set
            print(
                f"WARNING (z_Usermanager.py): User '{current_user_info_data.get('email')}' accessed admin via email check, but 'is_admin' DB flag is False/missing.")
            st.toast("Access granted via special override. Ensure DB admin flag is correctly set.", icon="⚠️")
else:  # user_info is not a dict, which shouldn't happen if login is correct
    print(f"ERROR (z_Usermanager.py): st.session_state.user_info is not a dictionary: {current_user_info_data}")

if not is_authorized_admin_flag:
    display_access_denied_content()


# ==================================
# USER MANAGEMENT DASHBOARD LAYOUT
# ==================================
def user_management_dashboard_layout():
    """Renders the main layout and functionality of the admin dashboard."""
    st.markdown(f"<h1 style='text-align: center; color: white;'>{PAGE_ICON} User Management Dashboard</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #E0E0E0;'>View and manage ScrubHub application users.</p>",
                unsafe_allow_html=True)

    # --- Main Interface Tabs ---
    tab1, tab2, tab3 = st.tabs(["Dashboard Overview", "User Management", "AI Usage Logs"])

    with tab1:
        st.subheader("Dashboard Overview")
        try:
            users_list_data: List[Dict[str, Any]] = get_all_users()
            df_users = pd.DataFrame(users_list_data)

            total_users = len(df_users)
            num_admins = df_users['is_admin'].sum() if 'is_admin' in df_users.columns else 0
            num_regular_users = total_users - num_admins

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Registered Users", total_users)
            col2.metric("Administrator Accounts", num_admins)
            col3.metric("Regular User Accounts", num_regular_users)

        except Exception as e:
            st.error(f"Could not load user data for dashboard: {e}")

    with tab2:
        st.subheader("Manage Registered Users")
        try:
            # We use the data fetched in the first tab if available
            if 'users_list_data' not in locals():
                users_list_data = get_all_users()
                df_users = pd.DataFrame(users_list_data)

            st.dataframe(df_users[['id', 'email', 'is_admin', 'created_at']], use_container_width=True, hide_index=True)
            st.caption(f"Total users: {len(df_users)}")

            st.markdown("---")
            st.subheader("User Actions")

            if users_list_data:
                user_email_options = [""] + sorted([user['email'] for user in users_list_data])
                selected_email = st.selectbox(
                    "Select User by Email to Manage:",
                    options=user_email_options,
                    format_func=lambda x: "Select a user..." if x == "" else x,
                )

                if selected_email:
                    selected_user = next((u for u in users_list_data if u['email'] == selected_email), None)
                    if selected_user:
                        user_id = selected_user['id']
                        st.markdown(f"**Managing:** {selected_user['email']} (ID: {user_id})")

                        # Expander for Resetting Password
                        with st.expander("Reset User Password"):
                            with st.form(key=f"reset_pass_{user_id}"):
                                new_pass = st.text_input("New Password", type="password")
                                conf_pass = st.text_input("Confirm New Password", type="password")
                                if st.form_submit_button("Reset Password"):
                                    if new_pass and new_pass == conf_pass and len(new_pass) >= 8:
                                        if reset_user_password(user_id, new_pass):
                                            st.success("Password reset successfully!")
                                        else:
                                            st.error("Failed to reset password.")
                                    else:
                                        st.warning("Ensure passwords match and are at least 8 characters long.")

                        # Expander for Changing Role
                        with st.expander("Change User Role"):
                            is_admin = selected_user.get('is_admin', False)
                            action_verb = "Revoke Admin Rights" if is_admin else "Grant Admin Rights"
                            st.write(f"This user is currently a **{'Administrator' if is_admin else 'Regular User'}**.")
                            if st.button(action_verb, use_container_width=True):
                                if toggle_admin_status(user_id, not is_admin):
                                    st.success("User role updated! Page will refresh.")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Failed to update user role.")

                        # Expander for Deleting User
                        with st.expander("Danger Zone"):
                            st.warning("WARNING: Deleting a user is permanent and cannot be undone.")
                            if st.button("Delete This User Permanently", use_container_width=True):
                                st.session_state[f'confirm_delete_{user_id}'] = True

                            if st.session_state.get(f'confirm_delete_{user_id}', False):
                                st.error("Are you absolutely sure?")
                                if st.button("Yes, I am sure. Delete now.", use_container_width=True):
                                    if delete_user(user_id):
                                        st.success("User deleted. Page will refresh.")
                                        del st.session_state[f'confirm_delete_{user_id}']
                                        time.sleep(1.5)
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete user.")
        except Exception as e:
            st.error(f"Failed to load user management tools: {e}")
            print(traceback.format_exc())

    with tab3:
        st.subheader("AI Usage Logs")
        try:
            logs_df = get_simulated_api_usage_logs()

            total_cost = logs_df['cost_usd'].sum()
            total_requests = len(logs_df)

            c1, c2 = st.columns(2)
            c1.metric("Total Simulated Cost (USD)", f"${total_cost:,.4f}")
            c2.metric("Total Simulated Requests", f"{total_requests}")

            st.write("Recent Simulated Activity:")
            display_df = logs_df.rename(columns={
                'email': 'User Email', 'model_used': 'Model', 'cost_usd': 'Cost (USD)', 'timestamp': 'Timestamp'
            })
            st.dataframe(display_df[['Timestamp', 'User Email', 'Model', 'Cost (USD)']], use_container_width=True,
                         hide_index=True)

        except Exception as e:
            st.error(f"Failed to generate or display simulated data: {e}")


# ==================================
# SCRIPT EXECUTION (Main call to layout function)
# ==================================
if __name__ == "__main__":
    # The main auth check is at the top. For direct execution (__main__),
    # we need to ensure st.session_state.user_info is set to an admin for testing.
    if 'user_info' not in st.session_state or st.session_state.user_info is None or \
            not st.session_state.user_info.get('is_admin'):

        # Only simulate if it's truly not set or not admin, to avoid loop with rerun
        if st.session_state.get('_admin_simulated_once_', False) is False:
            st.session_state.user_info = {
                "email": "whitney@datacleaner.com",
                "is_admin": True,
                "id": 0  # Dummy ID for testing purposes
            }
            st.session_state._admin_simulated_once_ = True  # Prevent rerun loop
            print("INFO (z_Usermanager.py __main__): Simulated admin user for direct testing. Rerunning.")
            st.rerun()
        # If already simulated and still not admin, something is wrong with the top check or logic.

    # If authorized (either by actual login or by __main__ simulation above that reran)
    if is_authorized_admin_flag:  # This flag is set by the top-level auth check
        user_management_dashboard_layout()
    else:
        # This else branch should ideally not be hit if rerun logic works,
        # as the top-level check would call display_access_denied_content() and st.stop().
        # But as a final fallback for direct run without proper session state setup:
        print("ERROR (z_Usermanager.py __main__): Not authorized to run dashboard even after simulation attempt.")
        if 'user_info' not in st.session_state or st.session_state.user_info is None:
            st.switch_page("a_Login.py")  # Send to login if no user info at all
        # display_access_denied_content() # Call it if user_info exists but not admin