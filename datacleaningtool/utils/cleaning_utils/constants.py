from typing import List, Dict

# Predefined cleaning options
CLEANING_STRATEGIES: List[str] = [
    'drop',
    'fill_with_mean',
    'fill_with_median',
    'fill_with_mode',
    'fill_with_value'
]

DATA_TYPES: List[str] = [
    'auto',
    'text',
    'number',
    'date',
    'category'
]

# Common data patterns
DATE_FORMATS: List[str] = [
    '%Y-%m-%d',
    '%m/%d/%Y',
    '%d/%m/%Y',
    '%Y%m%d',
    '%b %d, %Y'
]

# Error messages
ERROR_MESSAGES: Dict[str, str] = {
    'invalid_column': "Column not found in dataframe",
    'conversion_failed': "Could not convert values to target type",
    'empty_dataframe': "Dataframe contains no data"
}