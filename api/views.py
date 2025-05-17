from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from .models import ExcelFile
import pandas as pd
import json
import os
from django.conf import settings
import re
import numpy as np
from groq import Groq

# Global variable to store the loaded DataFrame
global_df = None
excel_file_path = None
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@api_view(['POST'])
def process_query(request):
    """
    Process a natural language query about the real estate data
    """
    global global_df

    # Check if data is loaded
    if global_df is None:
        # First try the default path mentioned in the code
        default_file_path = os.path.join(settings.MEDIA_ROOT, 'excel_files/real_estate.xlsx')

        # If the file doesn't exist at the default path, try directly in the media folder
        if not os.path.exists(default_file_path):
            default_file_path = os.path.join(settings.MEDIA_ROOT, 'real_estate.xlsx')

        # Check if the file exists at the new path
        if os.path.exists(default_file_path):
            try:
                global_df = pd.read_excel(default_file_path)
                print(f"Successfully loaded Excel file from {default_file_path}")
            except Exception as e:
                return JsonResponse({'error': f'Error loading Excel file: {str(e)}'}, status=500)
        else:
            return JsonResponse({
                'error': 'No data loaded. Excel file not found at expected locations. Please upload an Excel file first.'
            }, status=400)

    # Get the query from the request
    data = json.loads(request.body)
    query = data.get('query', '').lower()

    if not query:
        return JsonResponse({'error': 'No query provided'}, status=400)

    try:
        # Parse the query to identify areas, metrics, and time periods
        result = parse_and_process_query(query, global_df)
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({'error': f'Error processing query: {str(e)}'}, status=500)

def parse_and_process_query(query, df):
    """
    Parse the user query and extract relevant information from the DataFrame
    """
    # Convert column names to lowercase for easier matching
    df.columns = [col.lower() for col in df.columns]

    # Extract location names from the query
    # Get all unique locations from the data
    all_locations = df['final location'].unique()

    # Find mentioned locations in the query
    mentioned_locations = []
    for location in all_locations:
        if location.lower() in query.lower():
            mentioned_locations.append(location)

    # If no locations are found, return an error
    if not mentioned_locations:
        return {
            'error': 'No valid location found in your query. Try specifying a location like "Wakad, "Aundh", "Ambegaon Budruk" And Aundh.'
        }

    # Check if it's a comparison query
    is_comparison = len(mentioned_locations) > 1 or 'compare' in query.lower()

    # Filter data for the mentioned locations
    filtered_df = df[df['final location'].isin(mentioned_locations)]

    # Check for time range in the query
    time_range = extract_time_range(query)
    if time_range:
        filtered_df = filtered_df[filtered_df['year'].isin(time_range)]

    # Determine which metrics to analyze based on the query and comparison status
    metrics = determine_metrics(query, is_comparison)

    # Generate appropriate response based on the query type
    response = generate_response(filtered_df, mentioned_locations, metrics, is_comparison)

    return response

def extract_time_range(query):
    """
    Extract time range from the query
    """
    # Look for 'last X years' pattern
    last_years_match = re.search(r'last (\d+) years?', query)
    if last_years_match:
        num_years = int(last_years_match.group(1))
        current_year = 2025  # Since the current date is May 16, 2025
        return list(range(current_year - num_years, current_year + 1))

    # Look for year ranges like '2020 to 2023' or '2020-2023'
    year_range_match = re.search(r'(\d{4})(?:\s*[-to]+\s*)(\d{4})', query)
    if year_range_match:
        start_year = int(year_range_match.group(1))
        end_year = int(year_range_match.group(2))
        return list(range(start_year, end_year + 1))

    # Look for specific years
    years_match = re.findall(r'\b(20\d{2})\b', query)
    if years_match:
        return [int(year) for year in years_match]

    return None

def determine_metrics(query, is_comparison):
    """
    Determine which metrics to analyze based on the query and comparison status
    """
    # For comparison queries, only return 'demand' (total sold - igr)
    if is_comparison:
        return ['demand']

    # For non-comparison queries, determine metrics based on the query
    metrics = []

    if any(term in query for term in ['price', 'rate', 'cost']):
        metrics.append('price')

    if any(term in query for term in ['demand', 'sold', 'sales', 'popularity']):
        metrics.append('demand')

    if any(term in query for term in ['size', 'area', 'carpet', 'units']):
        metrics.append('size')

    # If no specific metrics are mentioned, include all for non-comparison queries
    if not metrics:
        metrics = ['price', 'demand', 'size']

    return metrics

def generate_response(df, locations, metrics, is_comparison):
    """
    Generate a response based on the filtered data
    """
    if df.empty:
        return {
            'error': 'No data found for the specified criteria'
        }

    # Prepare data for charts
    chart_data = prepare_chart_data(df, metrics)

    # Generate a text summary
    summary = generate_text_summary(df, locations, metrics, is_comparison)

    # Prepare table data - for comparison queries, only include relevant columns
    if is_comparison:
        table_data = df[['year', 'final location', 'total sold - igr']].to_dict('records')
    else:
        table_data = df.to_dict('records')

    return {
        'summary': summary,
        'chart_data': chart_data,
        'table_data': table_data,
        'metrics': metrics,
        'locations': locations,
    }

def prepare_chart_data(df, metrics):
    """
    Prepare data for charts based on the metrics
    """
    chart_data = {}

    # Group by year and location
    grouped = df.groupby(['year', 'final location'])

    # Price metrics
    if 'price' in metrics:
        price_data = []
        for (year, location), group in grouped:
            price_value = group['flat - weighted average rate'].mean()
            if not pd.isna(price_value):
                price_data.append({
                    'year': int(year),
                    'location': location,
                    'value': float(price_value)
                })
        chart_data['price'] = price_data

    # Demand metrics
    if 'demand' in metrics:
        demand_data = []
        for (year, location), group in grouped:
            demand_value = group['total sold - igr'].sum()
            if not pd.isna(demand_value):
                demand_data.append({
                    'year': int(year),
                    'location': location,
                    'value': float(demand_value)
                })
        chart_data['demand'] = demand_data

    # Size metrics
    if 'size' in metrics:
        size_data = []
        for (year, location), group in grouped:
            size_value = group['total units'].sum()
            if not pd.isna(size_value):
                size_data.append({
                    'year': int(year),
                    'location': location,
                    'value': float(size_value)
                })
        chart_data['size'] = size_data

    return chart_data

def generate_text_summary(df, locations, metrics, is_comparison):
    """
    Generate a text summary using Groq LLM
    """
    import json

    # Prepare a simple context about the data
    summary_context = {
        "locations": locations,
        "metrics": metrics,
        "is_comparison": is_comparison,
        "data": df.to_dict(orient="records")
    }

    system_prompt = (
        "You are a helpful assistant analyzing real estate market data. "
        "Generate a clear, concise, and human-readable summary based on the given context. "
        "Focus on prices, demand, and total units. Use bullet points and short paragraphs."
    )

    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this data:\n{json.dumps(summary_context)}"}
        ],
        temperature=0.7,
        max_completion_tokens=3000,
        top_p=1,
        stream=False,
    )

    return response.choices[0].message.content