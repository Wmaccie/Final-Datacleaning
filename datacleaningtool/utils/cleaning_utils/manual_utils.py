import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, Union, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import traceback


@dataclass
class CleaningSummary:
    original_shape: Tuple[int, int] = (0, 0);
    new_shape: Tuple[int, int] = (0, 0)
    rows_affected: int = 0;
    cells_standardized_to_nan: int = 0
    missing_values_filled: int = 0;
    missing_rows_dropped: int = 0
    columns_standardized: int = 0;
    type_conversions: int = 0
    action_taken: str = "N/A";
    details: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# VERNIEUWDE analyze_dataset FUNCTIE
# ============================================================================
def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Voert een uitgebreide, Pandas-compatibele analyse uit op de dataset.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {'error': 'Dataset is invalid or empty.'}
    try:
        df_analysis = df.copy()
        default_placeholders = ["", " ", "N/A", "NA", "NONE", "NULL", "Null", "-", "?", "onbekend", "MISSING", "nan",
                                "<NA>"]

        # Stap 1: Analyseer Missende Waarden (Robuust en Compatibel)
        for col in df_analysis.select_dtypes(include=['object', 'string']).columns:
            temp_col_as_str = df_analysis[col].astype(str)
            placeholders_lower = [str(p).lower() for p in default_placeholders if str(p).strip()]
            mask = temp_col_as_str.str.lower().str.strip().isin(placeholders_lower)
            df_analysis.loc[mask, col] = np.nan

        missing_info = {'total': int(df_analysis.isnull().sum().sum()),
                        'by_column': df_analysis.isnull().sum().to_dict()}

        # Stap 2: Analyseer Datatypes & Tekst Standaardisatie
        data_type_issues = []
        text_standardization_issues = []
        for col in df.columns:
            series = df[col]
            if series.dtype == 'object' and series.notna().any():
                numeric_series = pd.to_numeric(series, errors='coerce')
                if numeric_series.notna().sum() / len(series.dropna()) > 0.7:
                    data_type_issues.append(f"Column '{col}' seems to contain numbers but is stored as text.")
            if pd.api.types.is_string_dtype(series.dtype) or series.dtype == 'object':
                non_null_series = series.dropna().astype(str)
                if len(non_null_series) > 1:
                    if non_null_series.str.strip().ne(non_null_series).any():
                        text_standardization_issues.append(f"Column '{col}' has values with extra whitespace.")
                    if non_null_series.nunique() > non_null_series.str.lower().nunique():
                        text_standardization_issues.append(f"Column '{col}' has inconsistent casing.")

        return {
            'shape': df.shape, 'duplicates': {'count': int(df.duplicated().sum())},
            'missing': missing_info, 'data_type_issues': data_type_issues,
            'text_standardization_issues': text_standardization_issues
        }
    except Exception as e:
        print(f"Error in analyze_dataset: {e}\n{traceback.format_exc()}")
        return {'error': f'Analysis failed: {str(e)}'}

def remove_duplicates(
        df: pd.DataFrame,
        keep: Union[str, bool] = 'first',
        columns_to_consider: Optional[List[str]] = None,
        normalize_for_duplication_check: bool = False
) -> Tuple[pd.DataFrame, CleaningSummary, Optional[pd.DataFrame]]:
    summary = CleaningSummary(original_shape=df.shape, action_taken=f"Remove Duplicates (keep='{keep}')")
    trash_df: Optional[pd.DataFrame] = None
    if df.empty:
        summary.new_shape = df.shape
        return df.copy(), summary, None

    df_to_check = df.copy()

    if normalize_for_duplication_check:
        summary.details['normalization_applied'] = True
        cols_to_normalize = columns_to_consider if columns_to_consider else df_to_check.select_dtypes(
            include=['object', 'string']).columns
        for col in cols_to_normalize:
            if col in df_to_check.columns and (
                    pd.api.types.is_string_dtype(df_to_check[col].dtype) or pd.api.types.is_object_dtype(
                    df_to_check[col].dtype)):
                try:
                    df_to_check[col] = df_to_check[col].astype(str).str.lower().str.strip()
                except AttributeError:
                    pass

    if keep == 'first':
        duplicate_mask = df_to_check.duplicated(subset=columns_to_consider, keep='first')
    elif keep == 'last':
        duplicate_mask = df_to_check.duplicated(subset=columns_to_consider, keep='last')
    elif keep is False:
        duplicate_mask = df_to_check.duplicated(subset=columns_to_consider, keep=False)
    else:
        duplicate_mask = df_to_check.duplicated(subset=columns_to_consider, keep='first'); keep = 'first'

    if duplicate_mask.any(): trash_df = df[duplicate_mask].copy()

    df_cleaned = df.drop_duplicates(subset=columns_to_consider, keep=keep)

    summary.duplicates_removed = summary.original_shape[0] - df_cleaned.shape[0]
    summary.rows_affected = summary.duplicates_removed
    summary.new_shape = df_cleaned.shape
    summary.details['keep_strategy'] = str(keep)
    if columns_to_consider: summary.details['subset_columns'] = columns_to_consider
    if trash_df is not None and trash_df.empty: trash_df = None
    return df_cleaned, summary, trash_df


def handle_missing_values(
        df: pd.DataFrame,
        strategy: str = 'mode',
        fill_value: Any = None,
        column_to_process: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, CleaningSummary, Optional[pd.DataFrame]]:
    summary = CleaningSummary(original_shape=df.shape, action_taken=f"Handle Missing Values (Strategy: '{strategy}')")
    if df.empty or not column_to_process:
        summary.new_shape = df.shape
        return df.copy(), summary, None

    df_cleaned = df.copy()
    initial_missing = df_cleaned[column_to_process].isnull().sum().sum()

    if strategy == 'custom':
        df_cleaned[column_to_process] = df_cleaned[column_to_process].fillna(fill_value)
        summary.details['fill_value'] = fill_value

    elif strategy == 'mode':
        for col in column_to_process:
            if df_cleaned[col].isnull().any():
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col].fillna(mode_val[0], inplace=True)

    # --- NEW: Added 'median' strategy ---
    elif strategy == 'median':
        filled_cols = []
        skipped_cols = []
        for col in column_to_process:
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                filled_cols.append(col)
            else:
                skipped_cols.append(col)
        summary.details['median_filled_columns'] = filled_cols
        if skipped_cols:
            summary.details['skipped_non_numeric_columns'] = skipped_cols

    elif strategy == 'drop_row_any':
        rows_before = df_cleaned.shape[0]
        df_cleaned.dropna(subset=column_to_process, how='any', inplace=True)
        summary.missing_rows_dropped = rows_before - df_cleaned.shape[0]

    final_missing = df_cleaned[column_to_process].isnull().sum().sum()
    summary.missing_values_filled = int(initial_missing - final_missing)
    summary.rows_affected = summary.missing_values_filled + summary.missing_rows_dropped
    summary.new_shape = df_cleaned.shape
    summary.details['columns_processed'] = column_to_process

    # Returning None for trash_df for now, can be implemented later
    return df_cleaned, summary, None


def clean_data_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningSummary, Optional[pd.DataFrame]]:
    summary = CleaningSummary(original_shape=df.shape, action_taken="Optimize Data Types")
    if df.empty:
        summary.new_shape = df.shape
        return df.copy(), summary, None
    df_cleaned = df.copy()
    conversion_details = {}
    for col_name in df_cleaned.columns:
        original_type = df_cleaned[col_name].dtype
        converted_col = df_cleaned[col_name].convert_dtypes()
        if pd.api.types.is_object_dtype(converted_col.dtype):
            try:
                converted_col = pd.to_numeric(converted_col, errors='raise')
            except (ValueError, TypeError):
                try:
                    if pd.to_datetime(converted_col.dropna(), errors='coerce').notna().mean() > 0.7:
                        converted_col = pd.to_datetime(converted_col, errors='coerce')
                except Exception:
                    pass
        if original_type != converted_col.dtype:
            df_cleaned[col_name] = converted_col
            conversion_details[col_name] = f"{original_type} -> {converted_col.dtype}"
    summary.type_conversions = len(conversion_details)
    summary.new_shape = df_cleaned.shape
    summary.details['conversions_made'] = conversion_details
    return df_cleaned, summary, None

def standardize_values(
        df: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        text_to_lowercase: bool = False,
        text_to_uppercase: bool = False,
        strip_whitespace_all: bool = True,
        remove_chars_regex: Optional[str] = None,
        replace_chars_with: str = ''
) -> Tuple[pd.DataFrame, CleaningSummary, Optional[pd.DataFrame]]:
    summary = CleaningSummary(original_shape=df.shape, action_taken="Standardize Values")
    if df.empty:
        summary.new_shape = df.shape
        return df.copy(), summary, None

    df_cleaned = df.copy()
    cols_to_process = target_columns if target_columns else df_cleaned.select_dtypes(
        include=['object', 'string']).columns

    modified_cols_count = 0
    for col_name in cols_to_process:
        if col_name in df_cleaned.columns:
            original_series = df_cleaned[col_name].copy()
            # Convert to string type to use .str accessor safely
            current_col = df_cleaned[col_name].astype(pd.StringDtype())

            if strip_whitespace_all: current_col = current_col.str.strip()
            if text_to_lowercase:
                current_col = current_col.str.lower()
            elif text_to_uppercase:
                current_col = current_col.str.upper()

            # This part handles the new UI options "Remove punctuation" and "Remove digits"
            if remove_chars_regex:
                current_col = current_col.str.replace(remove_chars_regex, replace_chars_with, regex=True)

            # Only count as modified if the series has actually changed
            if not original_series.equals(current_col):
                df_cleaned[col_name] = current_col
                modified_cols_count += 1

    summary.columns_standardized = modified_cols_count
    summary.rows_affected = modified_cols_count  # Simplification, can be improved
    summary.new_shape = df_cleaned.shape
    summary.details = {
        'columns_processed': target_columns,
        'lowercase': text_to_lowercase,
        'uppercase': text_to_uppercase,
        'strip_whitespace': strip_whitespace_all,
        'removed_chars_pattern': remove_chars_regex
    }
    return df_cleaned, summary, None

# --- NEW: Backend function for renaming columns ---
def rename_columns(
        df: pd.DataFrame,
        rename_map: Dict[str, str]
) -> Tuple[pd.DataFrame, CleaningSummary, Optional[pd.DataFrame]]:
    summary = CleaningSummary(original_shape=df.shape, action_taken="Rename Columns")
    if df.empty or not rename_map:
        summary.new_shape = df.shape
        return df.copy(), summary, None

    df_cleaned = df.copy()

    # Safety check for duplicate new names
    new_names = list(rename_map.values())
    if len(new_names) != len(set(new_names)):
        raise ValueError("New column names must be unique.")

    df_cleaned.rename(columns=rename_map, inplace=True)

    summary.new_shape = df_cleaned.shape
    summary.details['columns_renamed'] = rename_map
    # In this case, rows_affected is not applicable, so we leave it at 0

    return df_cleaned, summary, None

# --- NEW FUNCTION ---
def normalize_dates(
        df: pd.DataFrame,
        target_format: str = '%Y-%m-%d',
        columns_to_process: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, CleaningSummary, Optional[pd.DataFrame]]:
    summary = CleaningSummary(original_shape=df.shape, action_taken="Normalize Dates")
    df_cleaned = df.copy()

    if columns_to_process is None:
        # If no columns are specified, intelligently find columns that look like dates
        columns_to_process = [col for col in df_cleaned.columns if
                              df_cleaned[col].astype(str).str.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').any()]

    if not columns_to_process:
        summary.details['info'] = "No date-like columns found to normalize."
        return df_cleaned, summary, None

    converted_cols = []
    for col in columns_to_process:
        if col in df_cleaned.columns:
            # Convert column to datetime objects, coercing errors to NaT (Not a Time)
            original_dates = df_cleaned[col].copy()
            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')

            # Format the valid dates into the target string format
            df_cleaned[col] = df_cleaned[col].dt.strftime(target_format)

            # If any changes were made, add to our report
            if not original_dates.equals(df_cleaned[col]):
                converted_cols.append(col)

    summary.new_shape = df_cleaned.shape
    summary.rows_affected = len(df_cleaned)  # This action affects all rows
    summary.details['columns_normalized'] = converted_cols
    summary.details['target_format'] = target_format

    return df_cleaned, summary, None

def full_clean(
        df: pd.DataFrame, **kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, CleaningSummary], Optional[Dict[str, Optional[pd.DataFrame]]]]:
    reports: Dict[str, Any] = {}
    trash: Dict[str, Optional[pd.DataFrame]] = {}
    df_current = df.copy()
    df_current, reports['remove_duplicates'], trash['removed_duplicates'] = remove_duplicates(df_current, **kwargs)
    df_current, reports['handle_missing_values'], trash['missing_values'] = handle_missing_values(df_current, **kwargs)
    df_current, reports['clean_data_types'], _ = clean_data_types(df_current)
    df_current, reports['standardize_values'], _ = standardize_values(df_current, **kwargs)
    reports['overall_summary'] = CleaningSummary(original_shape=df.shape, new_shape=df_current.shape,
                                                 action_taken="Full Clean")
    return df_current, reports, trash

def _is_name_column(column_name: str, series: pd.Series) -> bool:
    name_keywords = ['name', 'first', 'last', 'full', 'person', 'contact', 'customer', 'voornaam', 'achternaam']
    if any(kw in column_name.lower() for kw in name_keywords): return True
    if pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype):
        series_dropna = series.dropna()
        if len(series_dropna) < 5: return False
        sample = series_dropna.sample(n=min(20, len(series_dropna)), random_state=1)
        if sample.empty: return False
        name_like_count = sum(1 for x in sample if
                              isinstance(x, str) and x.replace(' ', '').replace('-', '').isalpha() and not any(
                                  char.isdigit() for char in x))
        if name_like_count / len(sample) > 0.6: return True
    return False

def _is_email_column(column_name: str, series: pd.Series) -> bool:
    email_keywords = ['email', 'e-mail', 'mail']
    if any(kw in column_name.lower() for kw in email_keywords): return True
    if pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype):
        series_dropna = series.dropna()
        if len(series_dropna) < 5: return False
        sample = series_dropna.sample(n=min(20, len(series_dropna)), random_state=1)
        if sample.empty: return False
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        match_count = sum(1 for x in sample if isinstance(x, str) and re.fullmatch(email_pattern, x.strip()))
        if match_count / len(sample) > 0.5: return True
    return False

def _is_date_column(column_name: str, series: pd.Series) -> bool:
    date_keywords = ['date', 'time', 'day', 'year', 'month', 'created_at', 'updated_at', 'timestamp', 'datum']
    if any(kw in column_name.lower() for kw in date_keywords): return True
    if pd.api.types.is_datetime64_any_dtype(series.dtype): return True
    if pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype):
        series_dropna = series.dropna()
        if len(series_dropna) < 5: return False
        sample = series_dropna.sample(n=min(20, len(series_dropna)), random_state=1)
        if sample.empty: return False
        try:
            converted_sample = pd.to_datetime(sample, errors='coerce')
            if converted_sample.notna().sum() / len(sample) > 0.6: return True
        except Exception:
            return False
    return False

def _convert_dates(series: pd.Series, target_format_strftime: str) -> pd.Series:
    datetime_series = pd.to_datetime(series, errors='coerce')
    if datetime_series.isnull().all(): return series
    try:
        not_nat_mask = datetime_series.notna()
        formatted_series = pd.Series(index=series.index, dtype=object)
        if not_nat_mask.any(): formatted_series.loc[not_nat_mask] = datetime_series[not_nat_mask].dt.strftime(
            target_format_strftime)
        formatted_series.loc[~not_nat_mask] = np.nan
        return formatted_series
    except Exception as e:
        print(f"Warning (manual_utils._convert_dates): Could not format date series: {e}")
        return series