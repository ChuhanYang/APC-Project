import os
import json
import glob
import pandas as pd
import numpy as np
# import time
import diskcache as dc  # For persistent caching
from geopy.geocoders import OpenCage


def generate_paths(vehicle_id: str, date: str) -> tuple:
    """
    Generates the data_folder and file_pattern given a vehicle_id and date.

    Parameters:
    vehicle_id (str): The ID of the vehicle (e.g., '406')
    date (str): The date in the format YYYYMMDD (e.g., '20240904')

    Returns:
    tuple: A tuple containing (data_folder, file_pattern)
    """

    # Construct the data_folder using the vehicle_id
    data_folder = f"\\\\srtlrv\\frveh\\V{vehicle_id.zfill(6)}"

    # Construct the file_pattern using the vehicle_id and date
    file_pattern = f"APC_{vehicle_id}_*_{date}_*.csv"

    return data_folder, file_pattern

def read_and_concatenate_csv(data_folder, file_pattern):
    """
    Reads and concatenates all CSV files in the specified folder matching the file pattern.

    Parameters:
    data_folder (str): The path to the folder containing the CSV files (e.g., data_folder = r"\\srtlrv\frveh\V000406")
    file_pattern (str): The pattern to match files (e.g., 'APC_406_60925_20240903_*.csv').

    Returns:
    pd.DataFrame: A concatenated DataFrame of all matching CSV files.
    """
    # Construct the full file pattern path
    file_pattern_full = os.path.join(data_folder, file_pattern)

    # Find all files that match the pattern
    csv_files = glob.glob(file_pattern_full)

    # If no files are found, raise an error
    if not csv_files:
        raise FileNotFoundError(f"No files found matching the pattern: {file_pattern_full}")

    # Read all CSV files and append to a list
    df_list = [pd.read_csv(file) for file in csv_files]

    # Concatenate all dataframes into one
    concatenated_df = pd.concat(df_list, ignore_index=True)

    return concatenated_df


# def compute_passenger_count(df, threshold_time='03:00:00', ensure_non_negative=True):
#     """
#     Computes the number of passengers on the train based on 'APC_Count_In' and 'APC_Count_Out',
#     and filters the data to start counting after a specified threshold time (e.g., 3 AM).
#
#     Parameters:
#     df (pd.DataFrame): The dataframe containing 'APC_Count_In', 'APC_Count_Out', 'Local_Time', and 'Local_Date'.
#     threshold_time (str): The time (in 'HH:MM:SS' format) after which to start the passenger count. Default is 3 AM.
#     ensure_non_negative (bool): Whether to ensure the passenger count doesn't go below zero. Default is True.
#
#     Returns:
#     pd.DataFrame: A dataframe with 'datetime' and 'passengers_on_train' columns.
#     """
#
#     # Aggregate the data by 'APC_Door_Nr', 'Local_Time', and 'Local_Date'
#     aggregated_df = df.groupby(
#         ['APC_Door_Nr', 'Local_Time', 'Local_Date'], as_index=False
#     ).agg({
#         'APC_Count_In': 'sum',
#         'APC_Count_Out': 'sum'
#     })
#
#     # Create a datetime column from 'Local_Date' and 'Local_Time'
#     aggregated_df['datetime'] = pd.to_datetime(aggregated_df['Local_Date'] + ' ' + aggregated_df['Local_Time'])
#
#     # Sort the DataFrame by the 'datetime' column
#     aggregated_df = aggregated_df.sort_values(by='datetime')
#
#     # Convert the threshold_time to a datetime.time object
#     threshold_time_obj = pd.to_datetime(threshold_time).time()
#
#     # Filter the DataFrame to only include times after the threshold time
#     aggregated_df_after_threshold = aggregated_df[aggregated_df['datetime'].dt.time >= threshold_time_obj].copy()
#
#     # Further aggregate by 'datetime'
#     aggregated_df_final = aggregated_df_after_threshold.groupby(['datetime'], as_index=False).agg({
#         'APC_Count_In': 'sum',
#         'APC_Count_Out': 'sum'
#     })
#
#     # Initialize a new column for the number of passengers on the train
#     aggregated_df_final['passengers_on_train'] = 0
#
#     # Initialize the current passenger count to 0
#     current_passenger_count = 0
#
#     # Group by date and compute the passenger count for each day
#     for date, group in aggregated_df_final.groupby(aggregated_df_final['datetime'].dt.date):
#         # Reset the passenger count at the start of the day
#         current_passenger_count = 0
#
#         # Iterate through the group and update the passenger count
#         for idx, row in group.iterrows():
#             # Update the current passenger count based on 'APC_Count_In' and 'APC_Count_Out'
#             current_passenger_count += row['APC_Count_In'] - row['APC_Count_Out']
#
#             # Ensure the passenger count doesn't go below zero, if specified by the user
#             if ensure_non_negative:
#                 current_passenger_count = max(0, current_passenger_count)
#
#             # Store the updated passenger count in the dataframe
#             aggregated_df_final.loc[idx, 'passengers_on_train'] = current_passenger_count
#
#     return aggregated_df_final

def parse_nmea_gprmc(nmea_string):
    """
    Parses an NMEA GPRMC string and extracts the latitude and longitude if the status is 'A'.
    Returns a dictionary with parsed information or None if parsing fails.
    """
    if pd.isna(nmea_string):
        return None

    try:
        # Split the NMEA string into its components
        parts = nmea_string.split(';')

        if len(parts) < 12:
            return None

        # Check if the status is 'A' (active data)
        status = parts[2]
        if status != 'A':
            return None

        # Extract latitude and longitude
        lat = parts[3]
        lat_dir = parts[4]
        lon = parts[5]
        lon_dir = parts[6]

        # Convert latitude and longitude to decimal degrees
        latitude = convert_to_decimal_degrees(lat, lat_dir)
        longitude = convert_to_decimal_degrees(lon, lon_dir)

        # Extract time and date (can be processed further if needed)
        time = parts[1]
        date = parts[9]

        return {
            'latitude': latitude,
            'longitude': longitude,
            'time': time,
            'date': date
        }

    except Exception as e:
        return None

def convert_to_decimal_degrees(value, direction):
    """
    Converts NMEA latitude/longitude format to decimal degrees.
    """
    if value == '' or pd.isna(value):
        return np.nan

    # Determine number of degrees based on the length of the value and direction
    if direction in ['N', 'S']:  # Latitude (ddmm.mmmm)
        degrees_len = 2
    else:  # Longitude (dddmm.mmmm)
        degrees_len = 3

    # Extract degrees and minutes
    degrees = float(value[:degrees_len])
    minutes = float(value[degrees_len:]) / 60

    # Convert to decimal degrees
    decimal_degrees = degrees + minutes

    # Apply negative sign for South or West directions
    if direction in ['S', 'W']:
        decimal_degrees *= -1

    return decimal_degrees


# def generate_location_dataframe(df1):
#     """
#     Extracts location information from the 'NMEA_GPRMC' column in df1, groups by unique datetime values,
#     and returns a DataFrame with train locations at each unique datetime (combination of Local_Time and Date columns).
#     """
#     # Create a list to hold the extracted data
#     location_data = []
#
#     # Ensure 'Local_Time' and 'Date' columns are of proper types
#     df1['Local_Time'] = pd.to_datetime(df1['Local_Time'], format='%H:%M:%S').dt.time
#     df1['Local_Date'] = pd.to_datetime(df1['Local_Date'], format='%m/%d/%Y').dt.date
#
#     # Combine 'Local_Time' and 'Date' columns into a single datetime column
#     df1['datetime'] = pd.to_datetime(df1['Local_Date'].astype(str) + ' ' + df1['Local_Time'].astype(str))
#
#     # Group by the 'datetime' column to ensure unique datetime values
#     df1_grouped = df1.groupby('datetime').first().reset_index()
#
#     # Iterate through each grouped row
#     for index, row in df1_grouped.iterrows():
#         nmea_string = row['NMEA_GPRMC']
#         parsed_data = parse_nmea_gprmc(nmea_string)
#
#         # If parsing was successful, append the data
#         if parsed_data:
#             location_data.append({
#                 'datetime': row['datetime'],  # Use the newly created 'datetime' column
#                 'latitude': parsed_data['latitude'],
#                 'longitude': parsed_data['longitude']
# #                'row_number': row.name  # Use the original row index after grouping
#             })
#
#     # Convert the list of dictionaries into a DataFrame
#     location_df = pd.DataFrame(location_data)
#
#     return location_df


def compute_passenger_count_and_location(df, threshold_time='03:00:00', ensure_non_negative=True):
    """
    Computes the number of passengers on the train, extracts train locations (latitude, longitude),
    and includes 'APC_Count_In' and 'APC_Count_Out' at each unique datetime.

    Parameters:
    df (pd.DataFrame): The dataframe containing 'APC_Count_In', 'APC_Count_Out', 'Local_Time', 'Local_Date', and 'NMEA_GPRMC'.
    threshold_time (str): The time (in 'HH:MM:SS' format) after which to start the passenger count. Default is 3 AM.
    ensure_non_negative (bool): Whether to ensure the passenger count doesn't go below zero. Default is True.

    Returns:
    pd.DataFrame: A dataframe with 'datetime', 'passengers_on_train', 'APC_Count_In', 'APC_Count_Out', 'latitude', and 'longitude'.
    """

    # # ---- Preprocessing: Combine Local Time and Date ----
    # df['Local_Time'] = pd.to_datetime(df['Local_Time'], format='%H:%M:%S').dt.time
    # df['Local_Date'] = pd.to_datetime(df['Local_Date'], format='%m/%d/%Y').dt.date
    df['Local_Time'] = pd.to_datetime(df['Local_Time'], errors='coerce').dt.time
    df['Local_Date'] = pd.to_datetime(df['Local_Date'], errors='coerce').dt.date
    df['datetime'] = pd.to_datetime(df['Local_Date'].astype(str) + ' ' + df['Local_Time'].astype(str))

    # Filter by threshold time
    threshold_time_obj = pd.to_datetime(threshold_time).time()
    df_filtered = df[df['datetime'].dt.time >= threshold_time_obj].copy()

    # ---- Aggregation by datetime ----
    # Aggregate by 'datetime' to sum 'APC_Count_In', 'APC_Count_Out' and get the first 'NMEA_GPRMC' value
    aggregated_df = df_filtered.groupby('datetime').agg({
        'APC_Count_In': 'sum',
        'APC_Count_Out': 'sum',
        'NMEA_GPRMC': 'first'
    }).reset_index()

    # ---- Passenger Count Calculation ----
    aggregated_df['passengers_on_train'] = 0  # Initialize passenger count
    current_passenger_count = 0  # Initialize running count

    for idx, row in aggregated_df.iterrows():
        current_passenger_count += row['APC_Count_In'] - row['APC_Count_Out']
        if ensure_non_negative:
            current_passenger_count = max(0, current_passenger_count)
        aggregated_df.at[idx, 'passengers_on_train'] = current_passenger_count

    # ---- Location Extraction from NMEA_GPRMC ----
    location_data = []
    for nmea_string in aggregated_df['NMEA_GPRMC']:
        parsed_data = parse_nmea_gprmc(nmea_string)
        if parsed_data:
            location_data.append({
                'latitude': parsed_data['latitude'],
                'longitude': parsed_data['longitude']
            })
        else:
            location_data.append({'latitude': None, 'longitude': None})

    # Convert the list of location data into a DataFrame and merge with the aggregated data
    location_df = pd.DataFrame(location_data)
    final_df = pd.concat([aggregated_df, location_df], axis=1)

    # Drop the NMEA_GPRMC column, as it's no longer needed
    final_df.drop(columns=['NMEA_GPRMC'], inplace=True)

    return final_df

# Access the OpenCage API key from the config file
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
api_key = config['api_keys']['opencage']

cache = dc.Cache("geocoding_cache")  # Store cached results in 'geocoding_cache' directory
geolocator = OpenCage(api_key)

def reverse_geocode_with_cache(lat, lon):
    """
    Reverse geocode using Nominatim with caching to avoid redundant API calls.

    Parameters:
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.

    Returns:
    str: The reverse-geocoded address or a message if not found.
    """
    # Create a unique cache key based on latitude and longitude
    cache_key = f"{lat},{lon}"

    # Check if the location is already in the cache
    if cache_key in cache:
        return cache[cache_key]  # Return the cached address

    # If not in cache, make an API call
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location:
            # Extract the first two parts of the address
            address = ', '.join(location.address.split(',')[:2])
            cache[cache_key] = address  # Cache the result
            return address
        else:
            return "Location not found"
    except Exception as e:
        return f"Error: {str(e)}"


def add_location_name(df):
    """
    Adds a 'location_name' column to the input DataFrame by reverse geocoding latitude and longitude using Nominatim.

    Parameters:
    df (pd.DataFrame): The DataFrame containing 'latitude' and 'longitude' columns.

    Returns:
    pd.DataFrame: The input DataFrame with an additional 'location_name' column.
    """
    # Apply reverse geocoding with caching to each row in the DataFrame
    df['location_name'] = df.apply(lambda row: reverse_geocode_with_cache(row['latitude'], row['longitude']), axis=1)

    # # Optionally, add a delay between API requests to avoid rate limits
    # time.sleep(1)

    return df


# Haversine formula to calculate distance between two lat-long points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


# For each row in result_df, find the nearest station from Station_GPS
def map_nearest_station(result_df, station_gps_df):
    # Initialize an empty column for Station_name
    result_df['Station_name'] = np.nan

    # Iterate over each row in result_df
    for idx, row in result_df.iterrows():
        train_lat = row['latitude']
        train_lon = row['longitude']

        # Calculate the distance to all stations
        station_gps_df['distance'] = station_gps_df.apply(
            lambda station: haversine(train_lat, train_lon, station['Latitude'], station['Longitude']), axis=1)

        # Find the nearest station
        nearest_station = station_gps_df.loc[station_gps_df['distance'].idxmin()]

        # Assign the nearest station's name to the result_df
        result_df.at[idx, 'Station_name'] = nearest_station['StationName']

    return result_df


def compute_door_count_and_location(df, threshold_time='03:00:00', ensure_non_negative=True, door_number=1):
    """
    Computes the sum of passengers boarding (APC_Count_In) and leaving (APC_Count_Out) the train at each door
    and aggregates by 'datetime' and 'APC_Door_Nr'. Also includes train locations (latitude, longitude) at each unique datetime.

    Parameters:
    df (pd.DataFrame): The dataframe containing 'APC_Count_In', 'APC_Count_Out', 'Local_Time', 'Local_Date', 'APC_Door_Nr', and 'NMEA_GPRMC'.
    threshold_time (str): The time (in 'HH:MM:SS' format) after which to start counting passengers. Default is 3 AM.
    ensure_non_negative (bool): Whether to ensure the passenger count doesn't go below zero. Default is True.
    door_number (int): The door number to filter by. Default is 1.

    Returns:
    pd.DataFrame: A dataframe with 'datetime', 'APC_Count_In', 'APC_Count_Out', 'latitude', and 'longitude' aggregated by door number.
    """

    # ---- Preprocessing: Combine Local Time and Date ----
    #df['Local_Time'] = pd.to_datetime(df['Local_Time'], format='%H:%M:%S').dt.time
    #df['Local_Date'] = pd.to_datetime(df['Local_Date'], format='%m/%d/%Y').dt.date
    df['Local_Time'] = pd.to_datetime(df['Local_Time'], errors='coerce').dt.time
    df['Local_Date'] = pd.to_datetime(df['Local_Date'], errors='coerce').dt.date
    df['datetime'] = pd.to_datetime(df['Local_Date'].astype(str) + ' ' + df['Local_Time'].astype(str))

    # Filter by threshold time
    threshold_time_obj = pd.to_datetime(threshold_time).time()
    df_filtered = df[(df['datetime'].dt.time >= threshold_time_obj) & (df['APC_Door_Nr'] == door_number)].copy()

    # ---- Aggregation by 'datetime' and 'APC_Door_Nr' ----
    # Aggregate by 'datetime' and 'APC_Door_Nr' to sum 'APC_Count_In', 'APC_Count_Out' and get the first 'NMEA_GPRMC' value
    aggregated_df = df_filtered.groupby(['datetime', 'APC_Door_Nr']).agg({
        'APC_Count_In': 'sum',
        'APC_Count_Out': 'sum',
        'NMEA_GPRMC': 'first'
    }).reset_index()

    # ---- Location Extraction from NMEA_GPRMC ----
    location_data = []
    for nmea_string in aggregated_df['NMEA_GPRMC']:
        parsed_data = parse_nmea_gprmc(nmea_string)
        if parsed_data:
            location_data.append({
                'latitude': parsed_data['latitude'],
                'longitude': parsed_data['longitude']
            })
        else:
            location_data.append({'latitude': None, 'longitude': None})

    # Convert the list of location data into a DataFrame and merge with the aggregated data
    location_df = pd.DataFrame(location_data)
    final_df = pd.concat([aggregated_df, location_df], axis=1)

    # Drop the NMEA_GPRMC column, as it's no longer needed
    final_df.drop(columns=['NMEA_GPRMC'], inplace=True)

    return final_df