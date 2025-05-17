import pandas as pd
import os
import numpy as np
from django.conf import settings

def load_excel_file(file_path=None):
    """
    Load and preprocess the Excel file

    Args:
        file_path: Path to the Excel file. If None, try to load the default file.

    Returns:
        Pandas DataFrame with preprocessed data
    """
    if file_path is None:
        # Try to load the default file
        file_path = os.path.join(settings.MEDIA_ROOT, 'excel_files/real_estate.xlsx')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found")

    # Load the Excel file
    df = pd.read_excel(file_path)

    # Clean the data
    df = preprocess_data(df)

    return df

def preprocess_data(df):
    """
    Preprocess the data for analysis

    Args:
        df: Pandas DataFrame with raw data

    Returns:
        Pandas DataFrame with preprocessed data
    """
    # Convert column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]

    # Handle missing values
    numeric_columns = [
        'flat - weighted average rate', 'office - weighted average rate',
        'others - weighted average rate', 'shop - weighted average rate',
        'total sold - igr', 'flat_sold - igr', 'office_sold - igr',
        'shop_sold - igr', 'total units', 'total carpet area supplied (sqft)'
    ]

    for col in numeric_columns:
        if col in df.columns:
            # Replace non-numeric values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Replace NaN with column median
            median = df[col].median()
            df[col].fillna(median, inplace=True)

    # Ensure year is an integer
    if 'year' in df.columns:
        df['year'] = df['year'].astype(int)

    # Clean location names
    if 'final location' in df.columns:
        df['final location'] = df['final location'].str.strip()

    return df

def filter_data_by_location(df, locations):
    """
    Filter the data by location

    Args:
        df: Pandas DataFrame with all data
        locations: List of location names to filter by

    Returns:
        Pandas DataFrame filtered to the specified locations
    """
    return df[df['final location'].isin(locations)]

def filter_data_by_year_range(df, start_year, end_year):
    """
    Filter the data by year range

    Args:
        df: Pandas DataFrame with all data
        start_year: Start year (inclusive)
        end_year: End year (inclusive)

    Returns:
        Pandas DataFrame filtered to the specified year range
    """
    return df[(df['year'] >= start_year) & (df['year'] <= end_year)]

def calculate_price_trends(df):
    """
    Calculate price trends for each location

    Args:
        df: Pandas DataFrame with filtered data

    Returns:
        Dictionary with price trends for each location
    """
    trends = {}

    # Group by location and year
    grouped = df.groupby(['final location', 'year'])

    for location in df['final location'].unique():
        location_df = df[df['final location'] == location]
        years = sorted(location_df['year'].unique())

        if len(years) < 2:
            continue

        price_data = []
        for year in years:
            year_df = location_df[location_df['year'] == year]
            price = year_df['flat - weighted average rate'].mean()
            price_data.append((year, price))

        # Calculate year-over-year changes
        changes = []
        for i in range(1, len(price_data)):
            prev_year, prev_price = price_data[i-1]
            curr_year, curr_price = price_data[i]

            if prev_price > 0:
                pct_change = ((curr_price - prev_price) / prev_price) * 100
                changes.append((curr_year, pct_change))

        trends[location] = {
            'prices': price_data,
            'changes': changes
        }

    return trends

def calculate_demand_trends(df):
    """
    Calculate demand trends for each location

    Args:
        df: Pandas DataFrame with filtered data

    Returns:
        Dictionary with demand trends for each location
    """
    trends = {}

    # Group by location and year
    grouped = df.groupby(['final location', 'year'])

    for location in df['final location'].unique():
        location_df = df[df['final location'] == location]
        years = sorted(location_df['year'].unique())

        if len(years) < 2:
            continue

        demand_data = []
        for year in years:
            year_df = location_df[location_df['year'] == year]
            demand = year_df['total sold - igr'].sum()
            demand_data.append((year, demand))

        # Calculate year-over-year changes
        changes = []
        for i in range(1, len(demand_data)):
            prev_year, prev_demand = demand_data[i-1]
            curr_year, curr_demand = demand_data[i]

            if prev_demand > 0:
                pct_change = ((curr_demand - prev_demand) / prev_demand) * 100
                changes.append((curr_year, pct_change))

        trends[location] = {
            'demand': demand_data,
            'changes': changes
        }

    return trends

def get_location_summary(df, location):
    """
    Get a summary of data for a specific location

    Args:
        df: Pandas DataFrame with all data
        location: Name of the location

    Returns:
        Dictionary with summary statistics for the location
    """
    location_df = df[df['final location'] == location]

    if location_df.empty:
        return None

    summary = {
        'location': location,
        'years_available': sorted(location_df['year'].unique()),
        'avg_flat_price': location_df['flat - weighted average rate'].mean(),
        'max_flat_price': location_df['flat - weighted average rate'].max(),
        'min_flat_price': location_df['flat - weighted average rate'].min(),
        'total_units_sold': location_df['total sold - igr'].sum(),
        'total_units_available': location_df['total units'].sum(),
    }

    return summary