from dash import Dash, html, dash_table, dcc, Input, Output, State
import plotly.graph_objects as go
from utils.helpers import generate_paths, read_and_concatenate_csv, compute_passenger_count
from utils.statistics import total_passenger_in, total_passenger_out, peak_passenger_on_board, \
    peak_passenger_in, peak_passenger_out, get_weather_statistics
import plotly.express as px
from utils.plotting import create_animated_trajectory
import pandas as pd

# Initialize the app
app = Dash()

# App layout
app.layout = html.Div([
    html.H1(children='SacRT APC Data Daily',
            style={
            'textAlign': 'center'
            }),

    html.Hr(),

    # Vehicle ID Dropdown
    html.Label("Select Vehicle ID:"),
    dcc.Dropdown(
        id='vehicle-id-dropdown',
        options=[{'label': f'Vehicle {i}', 'value': str(i)} for i in range(401, 421)],
        value='406',  # Default vehicle ID
        clearable=False
    ),

    # Date Input
    html.Label("Enter Date (YYYYMMDD):"),
    dcc.Input(
        id='date-input',
        type='text',
        value='20240904',  # Default date
        pattern=r'\d{8}',  # Date must be in format YYYYMMDD
        maxLength=8,
        placeholder="YYYYMMDD"
    ),

    # Button to load the data
    html.Button(id='load-data-btn', n_clicks=0, children='Load Data'),

    html.Hr(),

    # Warning message if no data is found
    html.Div(id='warning-message', style={'color': 'red'}),

    # Hidden div to store data
    dcc.Store(id='stored-data'),

    # DataTable to display raw data
    dash_table.DataTable(id='data-table', page_size=16),

    # Dropdown for selecting the plot type
    html.Label("Passenger Flow Bar Plot:"),
    dcc.Dropdown(
        id='plot-selection',
        options=[
            {'label': 'APC Count In', 'value': 'count_in'},
            {'label': 'APC Count Out', 'value': 'count_out'},
            {'label': 'Both', 'value': 'both'}
        ],
        value='both',  # Default selection
        clearable=True
    ),

    # Input for time interval in minutes
    html.Label("Select interval length (minutes):"),
    dcc.Input(
        id='interval-input',
        type='number',
        value=30,  # Default interval is 30 minutes
        min=1,  # Minimum interval is 1 minute
        step=1  # Step by 1 minute
    ),

    # Slider to select the time interval
    html.Label("Select start time:"),
    dcc.Slider(
        id='time-slider',
        min=0,
        max=0,  # This will be dynamically updated
        value=0,  # Default value
        marks={},  # This will be dynamically updated
        step=1
    ),

    # Graph to display the bar plot
    dcc.Graph(id='bar-graph'),

    # Graph to display the passengers on train over time
    dcc.Graph(id='passenger-line-graph'),

    # Statistics panel
    html.Div([
        html.H3("Daily Statistics"),
        html.P(id="total-in"),
        html.P(id="total-out"),
        html.P(id="peak-number"),
        html.P(id="peak-time"),
        html.P(id="peak-number-in"),
        html.P(id="peak-time-in"),
        html.P(id="peak-number-out"),
        html.P(id="peak-time-out")
    ], style={'border': '1px solid black', 'padding': '10px', 'width': '400px', 'margin': '20px'}),

    # Section for displaying weather statistics
    html.Div([
        html.H3("Weather Statistics (Sacramento, CA)"),
        html.P(id="weather-date"),
        html.P(id="temperature-avg"),
        html.P(id="temperature-max"),
        html.P(id="temperature-min"),
        html.P(id="precipitation"),
        html.P(id="wind-speed"),
        html.P(id="humidity")
    ], style={'border': '1px solid black', 'padding': '10px', 'width': '400px', 'margin': '20px'})

])

# Callback to load the data based on vehicle_id and date
@app.callback(
    [Output('stored-data', 'data'),
     Output('data-table', 'data'),
     Output('time-slider', 'max'),
     Output('time-slider', 'marks'),
     Output('warning-message', 'children')],
    [Input('load-data-btn', 'n_clicks')],
    [State('vehicle-id-dropdown', 'value'),
     State('date-input', 'value')]
)
def load_data(n_clicks, vehicle_id, date):
    if n_clicks > 0:
        # Generate the data_folder and file_pattern using vehicle_id and date
        data_folder, file_pattern = generate_paths(vehicle_id, date)

        # Read and concatenate CSV files
        df1 = read_and_concatenate_csv(data_folder, file_pattern)

        # If no data is found, display a warning
        if df1.empty:
            return {}, [], 0, {}, f"No data found for Vehicle {vehicle_id} on {date}. Please try another combination."

        # Compute passenger count
        result_df = compute_passenger_count(df1, threshold_time='03:00:00', ensure_non_negative=False)

        # Ensure the datetime column is of type datetime and sort the dataframe
        result_df['datetime'] = pd.to_datetime(result_df['datetime'])
        result_df = result_df.sort_values('datetime')

        # Generate the time intervals based on the data
        time_intervals = pd.date_range(start=result_df['datetime'].min(), end=result_df['datetime'].max(), freq='30min')

        # Prepare slider marks
        marks = {i: time_intervals[i].strftime('%H:%M') for i in range(0, len(time_intervals), max(1, len(time_intervals) // 10))}

        # Return the data and slider information, and reset the warning message
        return result_df.to_dict(), df1.to_dict('records'), len(time_intervals) - 1, marks, ''

    # Default return if no clicks yet
    return {}, [], 0, {}, ''

# Callback to update the bar plot and passenger line plot based on dropdown selection, slider, and interval input
@app.callback(
    [Output('bar-graph', 'figure'),
     Output('passenger-line-graph', 'figure')],
    [Input('plot-selection', 'value'),
     Input('time-slider', 'value'),
     Input('interval-input', 'value')],
    [State('stored-data', 'data')]
)
def update_charts(plot_selection, slider_value, interval_minutes, stored_data):
    if stored_data:
        # Reconstruct the result_df from the stored data
        result_df = pd.DataFrame.from_dict(stored_data)

        # Ensure the 'datetime' column is in datetime format
        result_df['datetime'] = pd.to_datetime(result_df['datetime'], errors='coerce')

        # Generate time intervals
        time_intervals = pd.date_range(start=result_df['datetime'].min(), end=result_df['datetime'].max(), freq=f'{interval_minutes}min')
        start_time = time_intervals[slider_value]
        end_time = start_time + pd.Timedelta(minutes=interval_minutes)

        # Filter data for bar and line plots
        filtered_df = result_df[(result_df['datetime'] >= start_time) & (result_df['datetime'] < end_time)]

        # Generate bar plot
        if plot_selection == 'count_in':
            fig_bar = px.bar(filtered_df, x='datetime', y='APC_Count_In', title=f'APC Count In from {start_time} to {end_time}')
        elif plot_selection == 'count_out':
            fig_bar = px.bar(filtered_df, x='datetime', y='APC_Count_Out', title=f'APC Count Out from {start_time} to {end_time}')
        else:
            fig_bar = go.Figure(data=[
                go.Bar(name='APC_Count_In', x=filtered_df['datetime'], y=filtered_df['APC_Count_In']),
                go.Bar(name='APC_Count_Out', x=filtered_df['datetime'], y=filtered_df['APC_Count_Out'])
            ])
            fig_bar.update_layout(barmode='group', title=f'APC Count In and Out from {start_time} to {end_time}')

        # Generate line plot for passengers on train
        fig_line = px.line(result_df, x='datetime', y='passengers_on_train', title='Number of Passengers on Train Over Time')

        return fig_bar, fig_line

    # Default empty charts if no data
    return {}, {}


# Callback to update statistics
@app.callback(
    [Output('total-in', 'children'),
     Output('total-out', 'children'),
     Output('peak-number', 'children'),
     Output('peak-time', 'children'),
     Output('peak-number-in', 'children'),
     Output('peak-time-in', 'children'),
    Output('peak-number-out', 'children'),
     Output('peak-time-out', 'children')
     ],
    [Input('stored-data', 'data')]
)
def update_statistics(stored_data):
    if stored_data:
        # Reconstruct the DataFrame from the stored data
        result_df = pd.DataFrame.from_dict(stored_data)

        # Convert 'datetime' back to proper datetime format
        result_df['datetime'] = pd.to_datetime(result_df['datetime'], errors='coerce')

        # Reset index to ensure it's integer-based
        result_df = result_df.reset_index(drop=True)

        # Compute statistics
        total_in = total_passenger_in(result_df)
        total_out = total_passenger_out(result_df)
        peak_number, peak_time = peak_passenger_on_board(result_df)
        peak_number_in, peak_time_in = peak_passenger_in(result_df)
        peak_number_out, peak_time_out = peak_passenger_out(result_df)

        # Format output
        total_in_str = f"Total Passengers In: {total_in}"
        total_out_str = f"Total Passengers Out: {total_out}"
        peak_number_str = f"Peak Passengers on board: {peak_number}"

        # Handle cases where peak_time is NaT
        if pd.isna(peak_time):
            peak_time_str = "Peak Time of passengers on board: No data"
        else:
            peak_time_str = f"Peak Time of passengers on board: {peak_time.strftime('%Y-%m-%d %H:%M:%S')}"

        peak_number_in_str = f"Peak number of passengers in (single time): {peak_number_in}"

        if pd.isna(peak_time_in):
            peak_time_in_str = "Peak Time of passengers in (single time): No data"
        else:
            peak_time_in_str = f"Peak Time of passengers in (single time): {peak_time_in.strftime('%Y-%m-%d %H:%M:%S')}"

        peak_number_out_str = f"Peak number of passengers Out (single time): {peak_number_out}"

        if pd.isna(peak_time_out):
            peak_time_out_str = "Peak Time of passengers Out (single time): No data"
        else:
            peak_time_out_str = f"Peak Time of passengers Out (single time): {peak_time_out.strftime('%Y-%m-%d %H:%M:%S')}"

        # Return the formatted strings for the statistics panel
        return total_in_str, total_out_str, peak_number_str, peak_time_str, \
            peak_number_in_str, peak_time_in_str, peak_number_out_str, peak_time_out_str

    # If no data is loaded, return empty strings
    return "", "", "", "", "", "", "", ""

# Callback to update weather statistics
@app.callback(
    [Output('weather-date', 'children'),
     Output('temperature-avg', 'children'),
     Output('temperature-max', 'children'),
     Output('temperature-min', 'children'),
     Output('precipitation', 'children'),
     Output('wind-speed', 'children'),
     Output('humidity', 'children')],
    [Input('date-input', 'value')]
)
def update_weather_statistics(date):
    # Convert the date to 'YYYY-MM-DD' format
    formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    # Fetch weather data for the specified date
    weather_data = get_weather_statistics(formatted_date)

    if "error" in weather_data:
        return [weather_data["error"]] * 7

    # Format the weather statistics
    date_str = f"Weather Date: {formatted_date}"
    temp_avg_str = f"Average Temperature: {weather_data['temperature_avg']} °C"
    temp_max_str = f"Max Temperature: {weather_data['temperature_max']} °C"
    temp_min_str = f"Min Temperature: {weather_data['temperature_min']} °C"
    precipitation_str = f"Precipitation: {weather_data['precipitation']} mm"
    wind_speed_str = f"Wind Speed: {weather_data['wind_speed']} km/h"
    humidity_str = f"Humidity: {weather_data['humidity']} %"

    return date_str, temp_avg_str, temp_max_str, temp_min_str, precipitation_str, wind_speed_str, humidity_str

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
