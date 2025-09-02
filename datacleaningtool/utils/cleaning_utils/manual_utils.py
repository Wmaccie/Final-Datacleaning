import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, Union, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import traceback
import requests
import streamlit as st
import pycountry

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
# --- NIEUWE HELPERFUNCTIE 1: Namen correct formatteren ---
def to_proper_case(name: str) -> str:
    """Converts a string to proper title case, handling edge cases like 'Mac Intosh'."""
    if not isinstance(name, str):
        return name

    # Een lijst van tussenvoegsels die klein moeten blijven (kan uitgebreid worden)
    lower_case_words = ['van', 'de', 'der', 'den', 'ten', 'ter', 'te']

    parts = name.strip().lower().split()
    capitalized_parts = []
    for part in parts:
        if part in lower_case_words:
            capitalized_parts.append(part)
        else:
            # Herkent "Mac" en "Mc" en zet de volgende letter in een hoofdletter
            if part.startswith(('mc', 'mac')):
                if len(part) > 2 and part[2].isalpha():
                    part = part[:2] + part[2].upper() + part[3:]
            capitalized_parts.append(part.capitalize())

    return ' '.join(capitalized_parts)


# --- NIEUWE HELPERFUNCTIE 2: Landcodes uitbreiden ---
def expand_country_code(code: str) -> str:
    """Converts a 2 or 3-letter country code to its full name."""
    if not isinstance(code, str) or len(code.strip()) > 3:
        return code  # Geef de originele waarde terug als het geen code is

    try:
        country = pycountry.countries.get(alpha_2=code.strip().upper())
        if country:
            return country.name
    except Exception:
        pass  # Ga door naar de volgende poging

    try:
        country = pycountry.countries.get(alpha_3=code.strip().upper())
        if country:
            return country.name
    except Exception:
        pass

    return code  # Geef de originele waarde terug als niets is gevonden


def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Voert een geavanceerde analyse uit die is afgestemd op de schoonmaaktools.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {'error': 'Dataset is invalid or empty.'}

    # Initialiseer de hoofdstructuur van de analyse
    analysis = {
        'shape': df.shape,
        'duplicates_found': int(df.duplicated().sum()),
        'missing_values_total': int(df.isnull().sum().sum()),
        'missing_values_by_column': df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        'column_analysis': {}
    }

    # Helperfuncties voor detectie binnen de analyse
    def _has_mixed_currencies(series: pd.Series) -> bool:
        if series.dtype != 'object' or series.isnull().all(): return False
        series_str = series.dropna().astype(str)
        found_symbols = set(re.findall(r'([€$]|EUR|USD|SRD)', ' '.join(series_str), re.IGNORECASE))
        return len(found_symbols) > 1

    def _is_country_code_col(series: pd.Series) -> bool:
        if series.dtype != 'object' or series.isnull().all(): return False
        sample = series.dropna().unique()
        short_text_sample = [s for s in sample if isinstance(s, str) and 2 <= len(s.strip()) <= 3 and s.isalpha()]
        if len(short_text_sample) < 3: return False
        valid_codes = 0
        for code in short_text_sample[:20]:
            try:
                if pycountry.countries.get(alpha_2=code.strip().upper()) or pycountry.countries.get(
                        alpha_3=code.strip().upper()):
                    valid_codes += 1
            except Exception:
                continue
        return (valid_codes / len(short_text_sample[:20])) > 0.5

    # Loop door elke kolom voor een gedetailleerd rapport
    for col in df.columns:
        series = df[col]
        col_report = {}

        # Basis-checks (whitespace, casing, etc.)
        if series.dtype == 'object':
            non_null_series = series.dropna().astype(str)
            if non_null_series.str.strip().ne(non_null_series).any():
                col_report['whitespace_issue'] = True
            if non_null_series.nunique() > non_null_series.str.lower().nunique():
                col_report['casing_issue'] = True

        # Potentiële datatype conversie check
        if series.dtype == 'object' and pd.to_numeric(series,
                                                      errors='coerce').notna().sum() / series.notna().count() > 0.7:
            col_report['potential_type_conversion'] = "Numeric (stored as text)"

        # --- NIEUWE, SLIMME DETECTIES ---
        # 1. Detectie van Datumkolommen
        try:
            if series.dtype == 'object' and pd.to_datetime(series, errors='coerce',
                                                           infer_datetime_format=True).notna().sum() / series.notna().count() > 0.6:
                col_report['potential_date_column'] = True
        except Exception:
            pass

        # 2. Detectie van Valutakolommen
        if _has_mixed_currencies(series):
            col_report['mixed_currency_issue'] = True

        # 3. Detectie van Landcodekolommen
        if _is_country_code_col(series):
            col_report['potential_country_code_column'] = True

        # 4. Detectie van Naamkolommen (voor Proper Case functie)
        if _is_name_column(col, series):
            col_report['potential_name_column'] = True

        if col_report:
            analysis['column_analysis'][col] = col_report

    return analysis


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
        replace_chars_with: str = '',
        # --- DE FIX: Voeg de nieuwe parameters toe aan de functiedefinitie ---
        apply_proper_case: bool = False,
        expand_countries: bool = False
) -> Tuple[pd.DataFrame, CleaningSummary, Optional[pd.DataFrame]]:
    summary = CleaningSummary(original_shape=df.shape, action_taken="Standardize Values")
    df_cleaned = df.copy()

    if not target_columns:
        cols_to_process = df_cleaned.select_dtypes(include=['object', 'string']).columns
    else:
        cols_to_process = target_columns

    modified_cols_count = 0
    for col_name in cols_to_process:
        if col_name in df_cleaned.columns:
            original_series = df_cleaned[col_name].copy()
            current_col = df_cleaned[col_name].astype(pd.StringDtype())

            if strip_whitespace_all:
                current_col = current_col.str.strip()

            if text_to_lowercase:
                current_col = current_col.str.lower()
            elif text_to_uppercase:
                current_col = current_col.str.upper()

            # --- NIEUWE LOGICA: Voer de 'proper case' functie uit ---
            if apply_proper_case:
                # We gebruiken .apply() om de to_proper_case helperfunctie op elke waarde toe te passen
                current_col = current_col.apply(lambda x: to_proper_case(x) if pd.notna(x) else x)

            # --- NIEUWE LOGICA: Voer de landcode-expansie uit ---
            if expand_countries:
                current_col = current_col.apply(lambda x: expand_country_code(x) if pd.notna(x) else x)

            if remove_chars_regex:
                current_col = current_col.str.replace(remove_chars_regex, replace_chars_with, regex=True)

            if not original_series.equals(current_col):
                df_cleaned[col_name] = current_col
                modified_cols_count += 1

    summary.columns_standardized = modified_cols_count
    summary.rows_affected = df_cleaned.shape[0]  # Actie kan elke rij beïnvloeden
    summary.details = {
        'columns_processed': target_columns,
        'lowercase': text_to_lowercase,
        'uppercase': text_to_uppercase,
        'strip_whitespace': strip_whitespace_all,
        'formatted_proper_names': apply_proper_case,
        'expanded_country_codes': expand_countries,
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


# --- NEW: Helper function to get exchange rates with caching ---
@st.cache_data(ttl=3600)  # Cache the results for 1 hour (3600 seconds)
def get_exchange_rates(api_key: str, base_currency: str = "EUR") -> Dict[str, float]:
    """Fetches latest exchange rates from the API and caches them."""
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        if data.get("result") == "success":
            print(f"INFO: Successfully fetched exchange rates with base {base_currency}.")
            return data["conversion_rates"]
        else:
            print(f"ERROR: API call was not successful. Response: {data}")
            return {}
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not fetch exchange rates: {e}")
        return {}


# --- NEW: Main currency conversion function ---
def convert_currency(
        df: pd.DataFrame,
        columns_to_process: List[str],
        target_currency: str,
        api_key: str
) -> Tuple[pd.DataFrame, CleaningSummary, Optional[pd.DataFrame]]:
    summary = CleaningSummary(original_shape=df.shape, action_taken=f"Convert Currency to {target_currency}")
    df_cleaned = df.copy()

    if not columns_to_process:
        raise ValueError("Please provide at least one column to process.")

    # --- Step 1: Get exchange rates ---
    # We use EUR as a stable base to convert from, then to the target.
    rates = get_exchange_rates(api_key, "EUR")
    if not rates:
        st.error("Could not retrieve live exchange rates. Please check your API key and internet connection.")
        return df, summary, None

    target_rate = rates.get(target_currency.upper())
    if not target_rate:
        st.error(f"Target currency '{target_currency}' not found in exchange rates.")
        return df, summary, None

    # --- Step 2: Define currency mappings and regex ---
    currency_map = {
        '€': 'EUR', 'EUR': 'EUR',
        '$': 'USD', 'USD': 'USD',
        'SRD': 'SRD'
    }
    # This regex captures currency symbols and the numeric value
    currency_pattern = re.compile(r'([€$]|EUR|USD|SRD)?\s*([\d,.-]+)')

    rows_affected = 0
    for col in columns_to_process:
        if col in df_cleaned.columns:
            original_series = df_cleaned[col].copy()

            def converter(value):
                if not isinstance(value, str):
                    return value  # Return non-strings as is

                match = currency_pattern.search(value)
                if not match:
                    return np.nan  # Not a recognizable currency format

                symbol, number_str = match.groups()
                source_currency = currency_map.get(symbol, 'EUR')  # Default to EUR if no symbol found

                try:
                    # Clean the number string
                    cleaned_number_str = number_str.replace('.', '', number_str.count('.') - 1).replace(',',
                                                                                                        '.').strip()
                    amount = float(cleaned_number_str)
                except (ValueError, TypeError):
                    return np.nan

                # Convert amount to EUR first, then to target currency
                amount_in_eur = amount / rates.get(source_currency, 1.0)
                converted_amount = amount_in_eur * target_rate
                return round(converted_amount, 2)

            df_cleaned[col] = df_cleaned[col].apply(converter)
            rows_affected += (original_series != df_cleaned[col]).sum()

    summary.new_shape = df_cleaned.shape
    summary.rows_affected = int(rows_affected)
    summary.details = {
        'columns_processed': columns_to_process,
        'target_currency': target_currency,
        'base_currency_for_rates': 'EUR'
    }

    return df_cleaned, summary, None