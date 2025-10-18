import pandas as pd
import pyarrow.parquet as pq
import pyarrow.parquet as pq

def extract_date_components(date_column):
    """
    Extract year, month number, and month abbreviation from a datetime Series as strings.
    Returns scalars if the Series has only one value.
    """
    date_column = pd.to_datetime(date_column)

    year_str = date_column.dt.year.astype(str)
    month_num_str = date_column.dt.month.astype(str).str.zfill(2)
    month_abbr_str = date_column.dt.strftime('%b').str.upper()

    # If only one value, return plain strings
    if len(date_column) == 1:
        return year_str.iloc[0], month_num_str.iloc[0], month_abbr_str.iloc[0]
    return year_str, month_num_str, month_abbr_str