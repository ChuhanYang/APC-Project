import pandas as pd
from meteostat import Point, Daily
from datetime import datetime

def total_passenger_in(df: pd.DataFrame) -> int:
    return df['APC_Count_In'].astype(int).sum()

def total_passenger_out(df: pd.DataFrame) -> int:
    return df['APC_Count_Out'].astype(int).sum()

def peak_passenger_on_board(df: pd.DataFrame) -> tuple:
    peak_number = df['passengers_on_train'].astype(int).max()
    peak_time = df['datetime'].iloc[df['passengers_on_train'].astype(int).idxmax()]
    return peak_number, peak_time

def peak_passenger_in(df: pd.DataFrame) -> tuple:
    peak_number = df['APC_Count_In'].astype(int).max()
    peak_time = df['datetime'].iloc[df['APC_Count_In'].astype(int).idxmax()]
    return peak_number, peak_time

def peak_passenger_out(df: pd.DataFrame) -> tuple:
    peak_number = df['APC_Count_Out'].astype(int).max()
    peak_time = df['datetime'].iloc[df['APC_Count_Out'].astype(int).idxmax()]
    return peak_number, peak_time


def get_weather_statistics(date: str) -> dict:
    """
    Fetches historical weather information for Sacramento, California on a specific date using Meteostat.

    Parameters:
        date (str): The date in the format 'YYYY-MM-DD'.

    Returns:
        dict: A dictionary containing weather information for Sacramento on the specified date.
    """
    # Coordinates for Sacramento, California
    latitude = 38.5757
    longitude = -121.2788

    # Convert the date string to a datetime object
    start = datetime.strptime(date, '%Y-%m-%d')
    end = start  # Meteostat allows date ranges, but we use a single day

    # Create a Point for Sacramento (latitude, longitude)
    location = Point(latitude, longitude)

    # Retrieve daily weather data for the specified location and date
    data = Daily(location, start, end)
    data = data.fetch()

    if data.empty:
        return {"error": "No weather data available for Sacramento on the specified date."}

    # Extract weather information
    weather_info = {
        "date": date,
        "temperature_avg": data['tavg'].iloc[0] if 'tavg' in data.columns else None,  # Average temperature
        "temperature_max": data['tmax'].iloc[0] if 'tmax' in data.columns else None,  # Max temperature
        "temperature_min": data['tmin'].iloc[0] if 'tmin' in data.columns else None,  # Min temperature
        "precipitation": data['prcp'].iloc[0] if 'prcp' in data.columns else None,  # Precipitation
        "wind_speed": data['wspd'].iloc[0] if 'wspd' in data.columns else None,  # Wind speed
        "humidity": data['rhum'].iloc[0] if 'rhum' in data.columns else None  # Humidity
    }

    return weather_info
