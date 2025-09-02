import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path
import traceback  # For detailed error logging
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from typing import Optional, Tuple, Dict, Any, List  # Ensure all are here

# --- Pad naar de assets map (consistent met d_Clean.py) ---
try:
    ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
except NameError:
    ASSETS_DIR = Path("assets")


# ==================================
# TYPE SAFETY HELPER FOR STANDALONE TESTING
# ==================================
def _ensure_arrow_safe_dtypes_for_review_page(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None: return None
    if not isinstance(df, pd.DataFrame):
        print(
            f"Warning (_ensure_arrow_safe_dtypes_for_review_page): Expected DataFrame, got {type(df)}. Returning as is.")
        return df

    df_copy = df.copy()
    # print(f"DEBUG (e_Datareview._ensure_arrow_safe_dtypes): Processing DataFrame with shape {df_copy.shape}. Initial dtypes:\n{df_copy.dtypes}")
    for col in df_copy.columns:
        original_dtype = df_copy[col].dtype
        if original_dtype == 'object':
            original_null_count = df_copy[col].isnull().sum()
            try:
                converted_numeric = pd.to_numeric(df_copy[col], errors='coerce')
                if converted_numeric.isnull().all() and not df_copy[col].isnull().all():
                    df_copy[col] = df_copy[col].astype(str)
                    continue
                newly_nulled_count = converted_numeric.isnull().sum() - original_null_count
                valid_data_before_coercion = len(df_copy[col]) - original_null_count
                if newly_nulled_count > 0 and valid_data_before_coercion > 0:
                    df_copy[col] = df_copy[col].astype(str)
                else:
                    df_copy[col] = converted_numeric
            except (ValueError, TypeError) as e_conv_rev_main:
                print(
                    f"DEBUG (e_Datareview): Fallback to string for column '{col}' due to conversion error: {e_conv_rev_main}")
                df_copy[col] = df_copy[col].astype(str)
            except Exception as e_other_rev_main:
                print(
                    f"DEBUG (e_Datareview): Fallback to string for column '{col}' due to an unexpected error: {e_other_rev_main}")
                df_copy[col] = df_copy[col].astype(str)
        if pd.api.types.is_integer_dtype(df_copy[col]) and df_copy[col].isnull().any():
            df_copy[col] = df_copy[col].astype(float)
    # print(f"DEBUG (e_Datareview._ensure_arrow_safe_dtypes): Finished processing. New dtypes:\n{df_copy.dtypes}")
    return df_copy


# ==================================
# FALLBACK DEFINITIONS
# ==================================
class FallbackPageConfigReview:
    PAGE_TITLE = "Data Review (Fallback)"
    PAGE_ICON = "📊"


# Probeer de gedeelde PageConfig te importeren, anders fallback
try:
    from utils.shared import PageConfig as ReviewPageConfig

    if not hasattr(ReviewPageConfig, 'PAGE_TITLE') or not hasattr(ReviewPageConfig, 'PAGE_ICON'):
        print("WARNING (e_Datareview): Imported PageConfig from utils.shared is missing attributes. Using fallback.")
        ReviewPageConfig = FallbackPageConfigReview
except ImportError:
    print("WARNING (e_Datareview): utils.shared.PageConfig not found. Using fallback PageConfig.")
    ReviewPageConfig = FallbackPageConfigReview
except Exception as e_imp_shared_rev_main:
    print(f"ERROR (e_Datareview): Importing utils.shared: {e_imp_shared_rev_main}. Using fallback PageConfig.")
    ReviewPageConfig = FallbackPageConfigReview

# ==================================
# PAGINA CONFIGURATIE - EERSTE STREAMLIT COMMANDO
# ==================================
try:
    st.set_page_config(
        page_title=getattr(ReviewPageConfig, 'PAGE_TITLE', "Data Review"),
        page_icon=getattr(ReviewPageConfig, 'PAGE_ICON', "📊"),
        layout="wide"
    )
except Exception as e_config_rev_main:
    st.set_page_config(page_title="Data Review - Config Error", page_icon="🔥", layout="wide")
    print(f"CRITICAL_ERROR (e_Datareview): Setting page config: {e_config_rev_main}")
    st.error(f"Page Configuration Error: {e_config_rev_main}. Defaulting to error page config.")


# ==================================
# STYLING EN ACHTERGROND FUNCTIE
# ==================================
def set_review_page_styling(background_image_path: Path, logo_image_path: Path):
    bg_image_css_val = "background-color: #0E0000;"
    if background_image_path.is_file():
        try:
            with open(background_image_path, "rb") as f_bg_rev_style:
                base64_bg_img = base64.b64encode(f_bg_rev_style.read()).decode()
            bg_image_css_val = f"""
                background-image: url("data:image/jpeg;base64,{base64_bg_img}");
                background-size: cover; background-position: center;
                background-repeat: no-repeat; background-attachment: fixed;
            """
        except Exception as e_bg_rev_styling:
            print(f"ERROR (e_Datareview): Loading background image '{background_image_path}': {e_bg_rev_styling}")
    else:
        print(f"WARNING (e_Datareview): Background image not found: '{background_image_path}'. Using fallback color.")

    # logo_image_path is passed but only checked, not used in CSS.
    # This might be flagged by a linter as "unused" if it doesn't see the print as usage.
    if not logo_image_path.is_file():
        print(f"WARNING (e_Datareview styling func): Logo file for styling check not found: '{logo_image_path}'")

    page_styling_css = f"""
        <style>
            .stApp {{ {bg_image_css_val} color: white !important; }}
            h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, [data-testid="stText"],
            .stMetric > label, .stMetric div[data-testid="stMetricValue"], .stMetric div[data-testid="stMetricDelta"] {{
                 color: white !important; 
            }}
            .stButton>button {{
                color: white !important; background-color: black !important;
                border: 1px solid #B71C1C !important; border-radius: 8px !important;
                padding: 10px 24px !important; font-weight: bold !important;
                transition: background-color 0.2s ease-in-out, transform 0.1s ease, box-shadow 0.2s ease !important;
                box-shadow: 0 2px 4px rgba(255,0,0,0.2);
            }}
            .stButton>button:hover {{
                background-color: #B71C1C !important; border-color: #FFCDD2 !important;
                color: white !important; transform: scale(1.03); box-shadow: 0 4px 8px rgba(255,0,0,0.3);
            }}
            .stButton>button:active {{ background-color: #7f0000 !important; transform: scale(0.98); }}
            .stButton>button[kind="primary"] {{
                background-color: #D32F2F !important; border-color: #FFCDD2 !important;
                box-shadow: 0 2px 4px rgba(255,255,255,0.2);
            }}
            .stButton>button[kind="primary"]:hover {{ background-color: #E57373 !important; border-color: white !important; }}
            .page-header-container {{ text-align: center; margin-bottom: 25px; }}
            .page-header-container h1 {{
                font-weight: 700; font-size: 3rem;
                text-shadow: 3px 3px 10px rgba(0,0,0,0.7); letter-spacing: 0.05em;
            }}
            .logo-container {{ display: flex; justify-content: center; align-items: center; padding-top: 25px; margin-bottom: 15px; }}
            .logo-image {{ width: 220px; max-width: 65%; height: auto; filter: drop-shadow(0px 0px 10px rgba(255, 100, 100, 0.5)); }}
            div[data-testid="stTabs"] button[data-baseweb="tab"] {{
                background-color: rgba(10,0,0,0.3); color: #E0E0E0; border-radius: 10px 10px 0 0;
                border-bottom: 2px solid transparent; padding: 12px 20px; font-weight: 500;
            }}
            div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
                background-color: rgba(176,0,0,0.5); color: white; font-weight: bold; border-bottom: 2px solid #FF5252;
            }}
            div[data-testid="stTabContent"] {{
                background-color: rgba(0,0,0,0.65); border-radius: 0 0 10px 10px; padding: 25px;
                border: 1px solid rgba(255,82,82,0.2);
            }}
            .stDataFrame {{ border: 1px solid rgba(255,82,82,0.3) !important; }}
            .stDataFrame table th {{ background-color: rgba(128,0,0,0.7) !important; color: white !important; font-weight: bold !important; }}
            .stMetric {{
                background-color: rgba(20,0,0,0.4); border-left: 5px solid #D32F2F;
                padding: 15px; border-radius: 8px;
            }}
            .stExpander header {{ background-color: rgba(100,0,0,0.4) !important; border-radius: 8px !important; color: white !important; }}
        </style>
    """
    try:
        st.markdown(page_styling_css, unsafe_allow_html=True)
    except Exception as e_style_rev_apply:
        print(f"ERROR (e_Datareview): Applying review page styling: {e_style_rev_apply}")
        st.error("Could not apply custom page styles. Default styles may be used.")


set_review_page_styling(ASSETS_DIR / "background_blue.jpg", ASSETS_DIR / "logo_wit.png")


# ==================================
# DATA QUALITY SCORE HELPER
# ==================================
def calculate_data_quality_score(df: Optional[pd.DataFrame]) -> float:  # Added Optional
    if not isinstance(df, pd.DataFrame) or df.empty:
        return 0.0
    try:
        total_cells = np.prod(df.shape)
        if total_cells == 0: return 0.0
        missing_cells = df.isnull().sum().sum()
        completeness_score = (1 - (missing_cells / total_cells)) * 100 if total_cells > 0 else 0

        unique_scores: List[float] = []
        for col_name in df.columns:
            col = df[col_name]
            if len(col) > 0:
                score = min((col.nunique() / len(col)) * 100, 100.0)  # Ensure float for min/max
                unique_scores.append(score)
        avg_uniqueness_score = np.mean(unique_scores) if unique_scores else 0.0

        valid_type_count = 0
        potential_numeric_date_cols = 0
        for col_name in df.columns:
            col = df[col_name]
            if col.dtype == 'object':
                try:
                    temp_series = pd.to_numeric(col, errors='coerce')
                    if pd.api.types.is_numeric_dtype(temp_series) and not np.isinf(temp_series.dropna()).any():
                        potential_numeric_date_cols += 1
                        valid_type_count += 1
                except Exception:  # Catch errors during to_numeric if col is highly unusual
                    pass
            elif pd.api.types.is_numeric_dtype(col):
                potential_numeric_date_cols += 1
                if not np.isinf(col.dropna()).any(): valid_type_count += 1
            elif pd.api.types.is_datetime64_any_dtype(col):  # Handles all datetime variations
                potential_numeric_date_cols += 1;
                valid_type_count += 1

        validity_score = (
                                     valid_type_count / potential_numeric_date_cols) * 100 if potential_numeric_date_cols > 0 else 75.0
        overall_score = (completeness_score * 0.5) + (validity_score * 0.3) + (avg_uniqueness_score * 0.2)
        return min(max(overall_score, 0.0), 100.0)  # Ensure float for min/max
    except Exception as e_score_rev_calc:
        print(f"Error (e_Datareview) calculating data quality score: {e_score_rev_calc}\n{traceback.format_exc()}")
        return 0.0


# ==================================
# PLOTTING HELPERS
# ==================================
def plot_distribution(df: pd.DataFrame, column_name: str) -> Optional[go.Figure]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Cannot plot distribution for empty or invalid DataFrame.")
        return None
    try:
        if column_name not in df.columns:
            st.warning(f"Column '{column_name}' not found for plotting distribution.")
            return None

        data_for_plot = df[column_name].copy()
        if data_for_plot.isnull().all():
            st.info(f"Column '{column_name}' contains only missing values. No distribution to plot.")
            return None

        if pd.api.types.is_numeric_dtype(data_for_plot.dtype):  # Check original dtype for numeric
            data_for_plot.dropna(inplace=True)
            data_for_plot = data_for_plot[~np.isinf(data_for_plot)]  # Remove infinities
        elif pd.api.types.is_object_dtype(data_for_plot.dtype) or pd.api.types.is_string_dtype(data_for_plot.dtype):
            # For object/string, try to convert to numeric for histogram, else Plotly handles as categorical
            try:
                numeric_converted = pd.to_numeric(data_for_plot, errors='raise')  # Try to convert
                # If successful, use numeric version after cleaning NaNs and Infs
                numeric_converted.dropna(inplace=True)
                numeric_converted = numeric_converted[~np.isinf(numeric_converted)]
                if not numeric_converted.empty:
                    data_for_plot = numeric_converted
                else:
                    data_for_plot = df[column_name].dropna().astype(
                        str)  # Fallback to original strings if conversion yields nothing
            except (ValueError, TypeError):  # If not convertible to numeric, use as strings (categorical)
                data_for_plot = df[column_name].dropna().astype(str)  # Ensure it's string for categorical
        else:  # For other types like boolean, datetime, just drop NaNs
            data_for_plot.dropna(inplace=True)

        if data_for_plot.empty:
            st.info(f"No valid data in '{column_name}' to plot distribution after pre-processing.")
            return None

        fig = px.histogram(x=data_for_plot, title=f"Distribution of {column_name}", color_discrete_sequence=['#D32F2F'])
        fig.update_layout(
            bargap=0.1, template="plotly_dark", title_font_color="white",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.2)',
            xaxis_title=str(column_name), yaxis_title="Frequency", font_color="white"  # Ensure column_name is str
        )
        return fig
    except Exception as e_plot_dist_rev_main:
        st.error(f"Error generating distribution plot for '{column_name}': {e_plot_dist_rev_main}")
        print(
            f"PLOT_ERROR (e_Datareview - Distribution for {column_name}): {e_plot_dist_rev_main}\n{traceback.format_exc()}")
        return None


def plot_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Cannot plot correlation heatmap for empty or invalid DataFrame.")
        return None
    try:
        numeric_df_cols = []
        for col_name in df.columns:  # Iterate by name to handle potential duplicate column names (though bad practice)
            col_data = df[col_name]
            if pd.api.types.is_numeric_dtype(col_data.dtype):
                temp_col = col_data[~np.isinf(col_data)].copy()  # Exclude infinities
                if not temp_col.isnull().all(): numeric_df_cols.append(temp_col)
            elif pd.api.types.is_object_dtype(col_data.dtype) or pd.api.types.is_string_dtype(col_data.dtype):
                try:
                    converted_col = pd.to_numeric(col_data, errors='coerce')
                    converted_col = converted_col[~np.isinf(converted_col)]
                    if not converted_col.isnull().all(): numeric_df_cols.append(converted_col)
                except Exception:
                    pass

        if not numeric_df_cols:
            st.info("No suitable numerical columns found after processing to generate a correlation heatmap.")
            return None

        numeric_df = pd.concat(numeric_df_cols, axis=1).dropna(axis=1, how='all')

        if numeric_df.shape[1] < 2:
            st.info(f"Not enough numerical columns (found {numeric_df.shape[1]}, need >= 2) for correlation heatmap.")
            return None

        corr_matrix = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=corr_matrix.columns.astype(str), y=corr_matrix.columns.astype(str),
            # Ensure str for labels
            colorscale='Reds', reversescale=True, zmin=-1, zmax=1
        ))
        fig.update_layout(
            title="Correlation Heatmap of Numerical Features", template="plotly_dark",
            title_font_color="white", paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)', font_color="white", height=600
        )
        return fig
    except Exception as e_corr_rev_main:
        st.error(f"Error generating correlation heatmap: {e_corr_rev_main}")
        print(f"PLOT_ERROR (e_Datareview - Correlation): {e_corr_rev_main}\n{traceback.format_exc()}")
        return None


def plot_missing_values_matrix(df: pd.DataFrame) -> Optional[go.Figure]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Cannot plot missing values matrix for empty or invalid DataFrame.")
        return None
    try:
        missing_matrix = df.isnull().astype(int)
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix.values, x=missing_matrix.columns.astype(str),  # Ensure str for labels
            y=[f"Row {i}" for i in range(len(df))],
            colorscale=[[0, 'rgba(200,0,0,0.3)'], [1, 'rgba(100,100,100,0.3)']], showscale=False
        ))
        fig.update_layout(
            title="Missing Values Matrix (Red = Value Present, Grey = Value Missing)", template="plotly_dark",
            title_font_color="white", paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)', font_color="white",
            height=max(400, min(len(df) * 10, 800)), yaxis_showticklabels=(len(df) <= 50)
        )
        fig.update_yaxes(autorange="reversed")
        return fig
    except Exception as e_missing_rev_main:
        st.error(f"Error generating missing values matrix: {e_missing_rev_main}")
        print(f"PLOT_ERROR (e_Datareview - Missing Matrix): {e_missing_rev_main}\n{traceback.format_exc()}")
        return None


# ==================================
# HOOFDPAGINA LAYOUT
# ==================================
def review_page_layout():
    logo_path = ASSETS_DIR / "logo_wit.png"
    if logo_path.is_file():
        try:
            with open(logo_path, "rb") as f_logo_rev_layout:
                logo_base64 = base64.b64encode(f_logo_rev_layout.read()).decode()
            st.markdown(
                f"""<div class="logo-container"><img src="data:image/png;base64,{logo_base64}" class="logo-image" alt="Logo"></div>""",
                unsafe_allow_html=True)
        except Exception as e_logo_display_rev_main:
            print(f"WARNING (e_Datareview): Logo display error: {e_logo_display_rev_main}")

    st.markdown("<div class='page-header-container'><h1>Data Review Dashboard</h1></div>", unsafe_allow_html=True)

    if 'df' not in st.session_state or not isinstance(st.session_state.df, pd.DataFrame):
        st.error("🚫 No valid data for review. Please process data on the Cleaning page.")
        if st.button("⬅️ Go Back to Data Cleaning", key="back_to_clean_if_no_curr_df_review_v3"):
            try:
                st.switch_page("pages/d_Clean.py")
            except Exception as e_nav_rev_main:
                st.warning(f"Navigation failed: {e_nav_rev_main}. Use sidebar.")
        st.stop()
    current_df: pd.DataFrame = st.session_state.df

    _temp_original_df_ss = st.session_state.get('original_df')
    original_df: pd.DataFrame
    if isinstance(_temp_original_df_ss, pd.DataFrame):
        original_df = _temp_original_df_ss
    else:
        original_df = pd.DataFrame()
        if _temp_original_df_ss is None and 'original_df' in st.session_state:
            print("DEBUG (e_Datareview): 'st.session_state.original_df' was None. Using an empty DataFrame.")
        elif 'original_df' not in st.session_state:
            print("DEBUG (e_Datareview): 'original_df' key not found. Using an empty DataFrame.")
        else:
            print(
                f"DEBUG (e_Datareview): 'original_df' was not a DataFrame (type: {type(_temp_original_df_ss)}). Using empty DataFrame.")

    quality_score_val = calculate_data_quality_score(current_df)
    st.session_state.data_quality_score = quality_score_val
    score_color = "red" if quality_score_val < 50 else ("orange" if quality_score_val < 75 else "green")

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3>Overall Data Quality Score</h3>
        <div style="background: conic-gradient({score_color} {quality_score_val:.1f}%, #333 {quality_score_val:.1f}%); border-radius: 50%; width: 150px; height: 150px; display: flex; align-items: center; justify-content: center; margin: auto; box-shadow: 0 0 15px rgba(255,255,255,0.2);">
            <p style="color: white; font-size: 2.5em; font-weight: bold; margin: 0;">{quality_score_val:.1f}%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_overview, tab_columns, tab_visuals, tab_compare = st.tabs([
        "📄 Data Overview", "📊 Column Deep Dive", "📈 Visual Explorer", "🆚 Comparison"
    ])

    with tab_overview:
        st.subheader("Cleaned Dataset Preview")
        if not current_df.empty:
            st.markdown(f"**Shape:** {current_df.shape[0]} rows, {current_df.shape[1]} columns")
            try:
                mem_usage = current_df.memory_usage(deep=True).sum() / (1024 * 1024)
                st.markdown(f"**Memory usage:** {mem_usage:.2f} MB")
            except Exception as e_mem_rev_main:
                st.caption(f"Memory usage calculation error: {e_mem_rev_main}")
            st.dataframe(current_df, height=400, use_container_width=True)  # Arrow error source if df not safe
            with st.expander("Basic Dataset Information (.info())", expanded=False):
                try:
                    buffer = StringIO();
                    current_df.info(buf=buffer);
                    st.text(buffer.getvalue())
                except Exception as e_info_rev_main:
                    st.error(f"Could not generate .info(): {e_info_rev_main}")
        else:
            st.info("Cleaned dataset is empty.")

    with tab_columns:
        st.subheader("Detailed Column Analysis")
        if current_df.empty:
            st.warning("Dataset empty. No columns to analyze.")
        else:
            col_search, col_selector_container = st.columns([1, 2])
            search_term = col_search.text_input("Search column name:", key="col_search_review_input_final")
            available_columns = [col for col in current_df.columns if
                                 search_term.lower() in col.lower()] if search_term else list(current_df.columns)

            if not available_columns:
                st.warning(f"No columns found matching '{search_term}'.")
            else:
                selected_column = col_selector_container.selectbox(
                    "Select a column for detailed analysis:", available_columns, key="col_select_review_dd_final"
                )
                if selected_column and selected_column in current_df.columns:
                    st.markdown(f"#### Analysis for: `{selected_column}`")
                    col_data = current_df[selected_column]
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    stat_col1.metric("Data Type", str(col_data.dtype))
                    missing_count = col_data.isnull().sum()
                    missing_percentage = (missing_count / len(col_data) * 100) if len(col_data) > 0 else 0.0
                    stat_col2.metric("Missing Values", f"{missing_count} ({missing_percentage:.1f}%)")
                    stat_col3.metric("Unique Values", col_data.nunique())

                    dist_fig_col_tab = plot_distribution(current_df, selected_column)
                    if dist_fig_col_tab:
                        st.plotly_chart(dist_fig_col_tab, use_container_width=True,
                                        key=f"dist_plot_tab_cols_{selected_column}_final")

                    with st.expander(f"Detailed Statistics for '{selected_column}'", expanded=False):
                        try:
                            col_data_non_null = col_data.dropna()
                            if col_data_non_null.empty:
                                st.info(f"Column '{selected_column}' effectively empty after dropping NaNs for stats.")
                            elif pd.api.types.is_numeric_dtype(col_data_non_null):
                                st.write(col_data.describe())
                            elif pd.api.types.is_datetime64_any_dtype(col_data_non_null):
                                st.write(col_data.describe(datetime_is_numeric=True))
                            else:
                                st.write(col_data.astype(str).describe(include='all'))
                                if col_data.nunique() < 50:
                                    st.write("Value Counts:")
                                    st.dataframe(col_data.value_counts(dropna=False).reset_index().rename(
                                        columns={'index': 'Value', 'count': 'Count'}))
                                else:
                                    st.info("Too many unique values (>50) to display all value counts here.")
                        except Exception as e_desc_rev_main:
                            st.error(f"Could not generate stats for {selected_column}: {e_desc_rev_main}")
                elif selected_column:
                    st.warning(f"Column '{selected_column}' not found. Please re-select.")

    with tab_visuals:
        st.subheader("Visual Data Explorer")
        if current_df.empty:
            st.info("Dataset empty. No visualizations.")
        else:
            plot_type = st.selectbox("Select visualization type:",
                                     ["Distribution of a Column", "Correlation Heatmap (Numeric Features)",
                                      "Missing Values Matrix"],
                                     key="viz_type_select_main_final")
            try:
                if plot_type == "Distribution of a Column":
                    if not current_df.columns.empty:
                        col_to_plot_viz = st.selectbox("Select column for distribution:", current_df.columns,
                                                       key="viz_dist_col_select_final")
                        if col_to_plot_viz:
                            fig_dist_main = plot_distribution(current_df, col_to_plot_viz)
                            if fig_dist_main: st.plotly_chart(fig_dist_main, use_container_width=True,
                                                              key=f"viz_dist_plot_main_{col_to_plot_viz}_final")
                    else:
                        st.info("No columns to plot.")
                elif plot_type == "Correlation Heatmap (Numeric Features)":
                    fig_corr_main = plot_correlation_heatmap(current_df)
                    if fig_corr_main: st.plotly_chart(fig_corr_main, use_container_width=True,
                                                      key="correlation_heatmap_main_final")
                elif plot_type == "Missing Values Matrix":
                    fig_missing_main = plot_missing_values_matrix(current_df)
                    if fig_missing_main: st.plotly_chart(fig_missing_main, use_container_width=True,
                                                         key="missing_values_matrix_main_final")
            except Exception as e_viz_rev_main:
                st.error(f"Error generating visualization: {e_viz_rev_main}")
                print(f"VIZ_TAB_ERROR (e_Datareview): {e_viz_rev_main}\n{traceback.format_exc()}")

    with tab_compare:
        st.subheader("Comparison: Cleaned Data vs. Original Data")
        if original_df.empty and current_df.empty:
            st.info("Both original and cleaned datasets are empty.")
        elif original_df.empty:
            st.info("Original dataset was empty/unavailable. Current cleaned data summary:")
            st.markdown(
                f"**Shape:** {current_df.shape[0]} R, {current_df.shape[1]} C. **Missing:** {current_df.isnull().sum().sum()}.")
        elif current_df.empty:
            st.info("Cleaned dataset is empty. Original dataset summary:")
            st.markdown(
                f"**Shape:** {original_df.shape[0]} R, {original_df.shape[1]} C. **Missing:** {original_df.isnull().sum().sum()}.")
        else:
            try:
                st.markdown("##### Key Differences:")
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                delta_rows = current_df.shape[0] - original_df.shape[0]
                delta_cols = current_df.shape[1] - original_df.shape[1]
                comp_col1.metric("Original Rows", original_df.shape[0], delta=f"{delta_rows:+}", delta_color="inverse")
                comp_col2.metric("Original Columns", original_df.shape[1], delta=f"{delta_cols:+}",
                                 delta_color="inverse")
                original_missing_total = original_df.isnull().sum().sum()
                cleaned_missing_total = current_df.isnull().sum().sum()
                delta_missing = cleaned_missing_total - original_missing_total
                comp_col3.metric("Total Missing Vals (Original)", original_missing_total, delta=f"{delta_missing:+}",
                                 delta_color="inverse")

                st.markdown("---")
                st.markdown("##### Column-wise Missing Value Change:")

                # Create a combined index of all columns from both dataframes
                all_cols_compare = original_df.columns.union(current_df.columns)
                original_missing_s = original_df.reindex(columns=all_cols_compare).isnull().sum()
                cleaned_missing_s = current_df.reindex(columns=all_cols_compare).isnull().sum()

                missing_comp_df = pd.DataFrame({
                    'Column': all_cols_compare,
                    'Original Missing': original_missing_s,
                    'Cleaned Missing': cleaned_missing_s
                }).reset_index(drop=True)
                missing_comp_df['Change'] = missing_comp_df['Cleaned Missing'] - missing_comp_df['Original Missing']
                st.dataframe(
                    missing_comp_df[missing_comp_df['Change'] != 0].sort_values(by='Change', na_position='last'),
                    use_container_width=True)

                with st.expander("Data Type Changes (for common or changed columns)", expanded=False):
                    type_changes_list: List[Dict[str, str]] = []
                    for col_tc in all_cols_compare:
                        orig_dtype_str = str(original_df[col_tc].dtype) if col_tc in original_df else "N/A (New Column)"
                        clean_dtype_str = str(
                            current_df[col_tc].dtype) if col_tc in current_df else "N/A (Column Dropped)"
                        if orig_dtype_str != clean_dtype_str:
                            type_changes_list.append(
                                {'Column': col_tc, 'Original Type': orig_dtype_str, 'Cleaned Type': clean_dtype_str})
                    if type_changes_list:
                        st.dataframe(pd.DataFrame(type_changes_list), use_container_width=True)
                    else:
                        st.info("No data type changes detected for common/changed columns.")
            except Exception as e_comp_rev_main:
                st.error(f"Error during comparison: {e_comp_rev_main}")
                print(f"COMPARE_TAB_ERROR (e_Datareview): {e_comp_rev_main}\n{traceback.format_exc()}")

    st.divider()
    nav_cols_bottom = st.columns([1, 2, 2, 1])
    with nav_cols_bottom[1]:
        if st.button("⬅️ Refine Cleaning (Back to Cleaner)", key="back_to_clean_btn_review_final_v3",
                     use_container_width=True):
            try:
                st.switch_page("pages/d_Clean.py")
            except Exception as e_nav_b_rev_main:
                st.warning(f"Navigation failed: {e_nav_b_rev_main}. Use sidebar.")
    with nav_cols_bottom[2]:
        if st.button("➡️ Proceed to Export Options", key="goto_export_btn_review_final_v3", type="primary",
                     use_container_width=True):
            try:
                st.switch_page("pages/f_Export.py")
            except Exception as e_nav_f_rev_main:
                st.error(f"Failed to navigate to Export: {e_nav_f_rev_main}. Ensure 'pages/f_Export.py' exists.")
                print(f"ERROR_NAVIGATION (e_Datareview to f_Export): {e_nav_f_rev_main}")


# ==================================
# SCRIPT UITVOERING (voor standalone testen)
# ==================================
if __name__ == "__main__":
    page_file_name_main = Path(__file__).name
    print(f"INFO ({page_file_name_main} __main__): Running review_page_layout in standalone mode.")

    if 'df' not in st.session_state or not isinstance(st.session_state.df, pd.DataFrame):
        print(f"INFO ({page_file_name_main} __main__): 'df' not in session_state. Creating sample data.")
        sample_data_cleaned = {
            'ID_Clean': list(range(1, 21)),  # Ensure list for pd.DataFrame
            'Product_Name_Clean': [f'Item_{i:02d}' for i in range(1, 21)],
            'Category_Clean': list(np.random.choice(['Electronics', 'Books', 'Home Goods', 'Apparel', None], size=20,
                                                    p=[0.2, 0.2, 0.2, 0.2, 0.2])),
            'Price_USD_Clean': list(np.round(np.random.uniform(10.0, 500.0, size=20), 2)),
            'Stock_Qty_Clean': list(np.random.randint(0, 100, size=20)),
            'Review_Date_Clean': pd.to_datetime(['2024-{:02d}-{:02d}'.format(m, d) for m, d in
                                                 zip(np.random.randint(1, 7, 20), np.random.randint(1, 29, 20))]),
            'Notes_Clean': ['Cleaned' if i % 2 == 0 else np.nan for i in range(20)],
            'Name': ['John Doe', 'BETHANY CALLAHAN', '3rd Person', 'Alice W.'] * 5,  # Arrow problematic
            'Join Date': ['01/01/2022', '6/2/2022', '12 Dec 2021', 'Mar 15 2023'] * 5,  # Arrow problematic
            'Address': ['123 Main St', '69490 Peter Canyon', '456 Oak Ave, Apt B', '789 Pine Ln, Suite 100'] * 5,
            # Arrow problematic
            'Bank Balance': ['USD 1,000.00', 'SRD 4,938.17', '500.50 EUR', '2000'] * 5,  # Arrow problematic
            'Mixed_Col_Test': [1, 'two', 3.0, 'BETHANY C', 5, None, '7', 8.0, 9, 10] * 2
        }
        st.session_state.df = pd.DataFrame(sample_data_cleaned)
        # Explicitly add some NaNs after DataFrame creation if np.nan in list was problematic
        st.session_state.df.loc[st.session_state.df.sample(frac=0.1).index, 'Notes_Clean'] = np.nan
        st.session_state.df.loc[st.session_state.df.sample(frac=0.05).index, 'Price_USD_Clean'] = np.nan

        print(
            f"INFO ({page_file_name_main} __main__): Sample 'df' created. Pre-safety dtypes:\n{st.session_state.df.dtypes}")
        st.session_state.df = _ensure_arrow_safe_dtypes_for_review_page(st.session_state.df)
        print(
            f"INFO ({page_file_name_main} __main__): Sample 'df' processed for Arrow safety. Post-safety dtypes:\n{st.session_state.df.dtypes if st.session_state.df is not None else 'None'}")

    if 'original_df' not in st.session_state or not isinstance(st.session_state.original_df, pd.DataFrame):
        if isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty:
            temp_orig_df_main = st.session_state.df.sample(frac=0.8,
                                                           random_state=1).copy()  # Use different random_state than other page if needed
            # Simulate some differences for original_df
            if 'Price_USD_Clean' in temp_orig_df_main.columns:
                temp_orig_df_main['Price_USD_Clean'] = temp_orig_df_main['Price_USD_Clean'] * np.random.uniform(0.7,
                                                                                                                1.1,
                                                                                                                size=len(
                                                                                                                    temp_orig_df_main))
            if 'Bank Balance' in temp_orig_df_main.columns:  # Make original bank balance different
                temp_orig_df_main['Bank Balance'] = ['$ ' + str(x * 0.8) for x in
                                                     np.random.randint(1000, 5000, size=len(temp_orig_df_main))]

            st.session_state.original_df = temp_orig_df_main
            print(
                f"INFO ({page_file_name_main} __main__): Sample 'original_df' created. Pre-safety dtypes:\n{st.session_state.original_df.dtypes}")
            st.session_state.original_df = _ensure_arrow_safe_dtypes_for_review_page(st.session_state.original_df)
            print(
                f"INFO ({page_file_name_main} __main__): Sample 'original_df' processed. Post-safety dtypes:\n{st.session_state.original_df.dtypes if st.session_state.original_df is not None else 'None'}")
        else:
            st.session_state.original_df = pd.DataFrame()
            print(f"INFO ({page_file_name_main} __main__): 'df' was not suitable for 'original_df'. Setting to empty.")

    if 'data_quality_score' not in st.session_state and isinstance(st.session_state.df, pd.DataFrame):
        st.session_state.data_quality_score = calculate_data_quality_score(st.session_state.df)

    review_page_layout()
    print(f"INFO ({page_file_name_main} __main__): review_page_layout() finished.")