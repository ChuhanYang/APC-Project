from dash import Dash, html, dash_table, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils.helpers import generate_paths, read_and_concatenate_csv, compute_passenger_count_and_location, \
    map_nearest_station, compute_door_count_and_location
from utils.statistics import total_passenger_in, total_passenger_out, peak_passenger_on_board, \
    peak_passenger_in, peak_passenger_out, get_weather_statistics
from utils.plotting import create_animated_trajectory, plot_passenger_door_flow

EXPLAINER = """This dashboard demo reads the daily APC data"""

# Initialize the app with the Minty theme
app = Dash(external_stylesheets=[dbc.themes.MINTY])

# Define the layout with tabs
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("SacRT APC Daily Dashboard", className='text-center'), width=12)]),
    dbc.Row(dbc.Col(html.Hr(), width=12)),
    dcc.Markdown(EXPLAINER),

    dbc.Tabs([
        # Tab 1: APC Table
        dbc.Tab(label="APC Table", children=[
            html.Div([
                html.H2("APC Table", className='text-center'),
                html.Label("Select Vehicle ID:"),
                dcc.Dropdown(
                    id='vehicle-id-dropdown',
                    options=[{'label': f'Vehicle {i}', 'value': str(i)} for i in range(401, 421)],
                    value='406',  # Default vehicle ID
                    clearable=False
                ),
                html.Label("Enter Date (YYYYMMDD):"),
                dcc.Input(
                    id='date-input',
                    type='text',
                    value='20240904',  # Default date
                    pattern=r'\d{8}',  # Date must be in format YYYYMMDD
                    maxLength=8,
                    placeholder="YYYYMMDD"
                ),
                html.Br(),
                html.Button(id='load-data-btn', n_clicks=0, children='Load Data', className='btn btn-primary'),
                html.Div(id='warning-message', style={'color': 'red'}),
                dcc.Store(id='stored-data'),
                dash_table.DataTable(id='data-table', page_size=16)
            ])
        ]),

        # Tab 2: Visualization
        dbc.Tab(label="Visualization", children=[
            html.Div([
                html.H2("Visualization", className='text-center'),
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
                html.Label("Select interval length (minutes):"),
                dcc.Input(
                    id='interval-input',
                    type='number',
                    value=30,  # Default interval is 30 minutes
                    min=1,  # Minimum interval is 1 minute
                    step=1
                ),
                html.Label("Select start time:"),
                dcc.Slider(
                    id='time-slider',
                    min=0,
                    max=0,  # This will be dynamically updated
                    value=0,  # Default value
                    marks={},  # This will be dynamically updated
                    step=1
                ),
                dcc.Graph(id='bar-graph'),
                dcc.Graph(id='passenger-line-graph'),
                html.Label("Enter Door Number:"),
                dcc.Input(id='door-number-input', type='number', value=1, min=1, max=8, step=1),
                dcc.Graph(id='door-passenger-line-graph')
            ])
        ]),

        # Tab 2.5: Animation
        dbc.Tab(label="Animation", children=[
            html.Div([
                html.H2("Historical trajectories", className='text-center'),
                html.Label("Historical trajectories:"),
                dcc.Graph(id='trajectory-map')
            ])
        ]),

        # Tab 3: Passenger Scatter Map
        dbc.Tab(label="Distribution", children=[
            html.Div([
                html.H2("Passenger Distribution Map", className='text-center'),
                html.Label("Enter Start Time (HH:MM):"),
                dcc.Input(
                    id='start-time-input',
                    type='text',
                    value='07:00',  # Default start time
                    maxLength=5,
                    placeholder="HH:MM"
                ),
                html.Label("Enter End Time (HH:MM):"),
                dcc.Input(
                    id='end-time-input',
                    type='text',
                    value='08:00',  # Default end time
                    maxLength=5,
                    placeholder="HH:MM"
                ),
                html.Button(id='filter-time-btn', n_clicks=0, children='Select Time Slot', className='btn btn-primary'),
                dcc.Graph(id='scatter-map')
            ])
        ]),

        # Tab 4: Statistics Reports
        dbc.Tab(label="Statistics Reports", children=[
            html.Div([
                html.H2("Statistics Reports", className='text-center'),

                dbc.Tabs([
                    # Daily Statistics Tab
                    dbc.Tab(label="Daily Statistics", children=[
                        dbc.Row([
                            # Card for Total Passengers In and Out
                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H5("Total Passengers In", className="card-title"),
                                    html.P(id="total-in", className="card-text"),
                                ])
                            ], color="primary", inverse=True)),

                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H5("Total Passengers Out", className="card-title"),
                                    html.P(id="total-out", className="card-text"),
                                ])
                            ], color="primary", inverse=True)),
                        ], className="mb-4"),

                        # Row for Peak Passengers information
                        dbc.Row([
                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H5("Peak Passengers On Board", className="card-title"),
                                    html.P(id="peak-number", className="card-text"),
                                    html.P(id="peak-time", className="card-text"),
                                ])
                            ], color="info", inverse=True)),

                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H5("Peak Passengers In", className="card-title"),
                                    html.P(id="peak-number-in", className="card-text"),
                                    html.P(id="peak-time-in", className="card-text"),
                                ])
                            ], color="info", inverse=True)),

                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H5("Peak Passengers Out", className="card-title"),
                                    html.P(id="peak-number-out", className="card-text"),
                                    html.P(id="peak-time-out", className="card-text"),
                                ])
                            ], color="info", inverse=True)),
                        ])
                    ]),

                    # Weather Statistics Tab
                    dbc.Tab(label="Weather Statistics", children=[
                        dbc.Row([
                            # Weather Date and Temperature
                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H5("Weather Date", className="card-title"),
                                    html.P(id="weather-date", className="card-text"),
                                ])
                            ], color="warning", inverse=True)),

                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H5("Average Temperature", className="card-title"),
                                    html.P(id="temperature-avg", className="card-text"),
                                    html.P(id="temperature-max", className="card-text"),
                                    html.P(id="temperature-min", className="card-text"),
                                ])
                            ], color="warning", inverse=True)),
                        ], className="mb-4"),

                        # Precipitation, Wind Speed, Humidity
                        dbc.Row([
                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H5("Precipitation", className="card-title"),
                                    html.P(id="precipitation", className="card-text"),
                                ])
                            ], color="secondary", inverse=True)),

                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H5("Wind Speed", className="card-title"),
                                    html.P(id="wind-speed", className="card-text"),
                                ])
                            ], color="secondary", inverse=True)),

                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H5("Humidity", className="card-title"),
                                    html.P(id="humidity", className="card-text"),
                                ])
                            ], color="secondary", inverse=True)),
                        ])
                    ]),

                    # New Pie Charts Tab
                    dbc.Tab(label="Pie Charts", children=[
                        html.Div([
                            # User input for the pie threshold
                            html.Label("Set Pie Threshold (Station whose value lower than threshold will be grouped into 'Other Stations'):"),
                            dcc.Input(
                                id='pie-threshold-input',
                                type='number',
                                value=10,  # Default value
                                min=1,
                                step=1,
                                placeholder="Enter threshold"
                            ),
                        ], style={'margin-bottom': '20px'}),

                        # Row to display pie charts
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='pie-in-chart'), width=6),
                            dbc.Col(dcc.Graph(id='pie-out-chart'), width=6)
                        ])
                    ])
                ])
            ])
        ])

    ])
], fluid=True)

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
        data_folder, file_pattern = generate_paths(vehicle_id, date)
        df1 = read_and_concatenate_csv(data_folder, file_pattern)
        if df1.empty:
            return {}, {}, [], 0, {}, f"No data found for Vehicle {vehicle_id} on {date}. Please try another combination."

        result_df = compute_passenger_count_and_location(df1, threshold_time='03:00:00', ensure_non_negative=False)
        result_df['datetime'] = pd.to_datetime(result_df['datetime'], errors='coerce')
        result_df = result_df.dropna(subset=['datetime'])

        Station_GPS = pd.read_csv('data/Station_GPS.csv')
        result_df = map_nearest_station(result_df.dropna(), Station_GPS)
        if result_df.empty:
            return {}, {}, [], 0, {}, f"No valid datetime entries for Vehicle {vehicle_id} on {date}. Please try another combination."

        result_df = result_df.sort_values('datetime')
        time_intervals = pd.date_range(start=result_df['datetime'].min(), end=result_df['datetime'].max(), freq='30min')
        marks = {i: time_intervals[i].strftime('%H:%M') for i in
                 range(0, len(time_intervals), max(1, len(time_intervals) // 10))}
        return result_df.to_dict(), df1.to_dict('records'), len(time_intervals) - 1, marks, ''

    return {}, [], 0, {}, ''  # Default return if no clicks yet

# Callback to update the bar plot and passenger line plot based on dropdown selection, slider, and interval input
@app.callback(
    Output('bar-graph', 'figure'),
    [Input('plot-selection', 'value'),
     Input('time-slider', 'value'),
     Input('interval-input', 'value')],
    [State('stored-data', 'data')],
    prevent_initial_call=True
)
def update_bar_charts(plot_selection, slider_value, interval_minutes, stored_data):
    if stored_data:
        # Reconstruct the result_df from the stored data
        result_df = pd.DataFrame.from_dict(stored_data)

        # Ensure the 'datetime' column is in datetime format
        result_df['datetime'] = pd.to_datetime(result_df['datetime'], errors='coerce')

        # Generate time intervals
        #time_intervals = pd.date_range(start=result_df['datetime'].min(), end=result_df['datetime'].max(), freq=f'{interval_minutes}min')
        time_intervals = pd.date_range(start=result_df['datetime'].min(), end=result_df['datetime'].max(), freq='30min')
        start_time = time_intervals[slider_value]
        end_time = start_time + pd.Timedelta(minutes=interval_minutes)

        # Filter data for bar and line plots
        filtered_df = result_df[(result_df['datetime'] >= start_time) & (result_df['datetime'] < end_time)]

        # Generate bar plot
        if plot_selection == 'count_in':
            fig_bar = px.bar(filtered_df, x='datetime', y='APC_Count_In', title=f'APC Count In from {start_time} to {end_time}',
                             hover_data={'Station_name': True})
        elif plot_selection == 'count_out':
            fig_bar = px.bar(filtered_df, x='datetime', y='APC_Count_Out', title=f'APC Count Out from {start_time} to {end_time}',
                             hover_data={'Station_name': True})
        else:
            fig_bar = go.Figure(data=[
                go.Bar(name='APC_Count_In', x=filtered_df['datetime'], y=filtered_df['APC_Count_In'],
                       customdata=filtered_df[['latitude', 'longitude','Station_name']],  # Attach latitude and longitude
                       hovertemplate=(
                               'Date: %{x}<br>' +
                               'Count In: %{y}<br>' +
                               'Latitude: %{customdata[0]}<br>' +
                               'Longitude: %{customdata[1]}<br>' +
                               'Station Name: %{customdata[2]}<br>'
                       )
                       ),
                go.Bar(name='APC_Count_Out', x=filtered_df['datetime'], y=filtered_df['APC_Count_Out'],
                       customdata=filtered_df[['latitude', 'longitude','Station_name']],  # Attach latitude and longitude
                       hovertemplate=(
                               'Date: %{x}<br>' +
                               'Count Out: %{y}<br>' +
                               'Latitude: %{customdata[0]}<br>' +
                               'Longitude: %{customdata[1]}<br>' +
                               'Station Name: %{customdata[2]}<br>'
                       )
                       )
            ])
            fig_bar.update_layout(barmode='group', title=f'APC Count In and Out from {start_time} to {end_time}')

        return fig_bar

    # Default empty charts if no data
    return go.Figure()

# Callback to update the line plot
@app.callback(
    Output('passenger-line-graph', 'figure'),
    Input('time-slider', 'value'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def update_line_charts(slider_value, stored_data):
    if slider_value is not None and stored_data:
        # Load and preprocess the data
        result_df = pd.DataFrame.from_dict(stored_data)
        result_df['datetime'] = pd.to_datetime(result_df['datetime'], errors='coerce')

        # Reverse geocode and add location names only for new locations (optional)
        # result_df = add_location_name(result_df)

        # Generate the line plot with hover data
        fig_line = px.line(
            result_df,
            x='datetime',
            y='passengers_on_train',
            title='Number of Passengers on Train Over Time',
            hover_data={
                'longitude': True,
                'latitude': True,
            #    'location_name': True,
                'Station_name': True
            }
        )
        return fig_line

    # Return an empty figure if no data
    return go.Figure()


@app.callback(
    Output('door-passenger-line-graph', 'figure'),
    [Input('door-number-input', 'value')],
    [State('data-table', 'data')],
    prevent_initial_call=True
)
def update_door_passenger_flow(door_number, data_table):
    if data_table:
        # Convert the 'data-table' content back into a DataFrame
        df1 = pd.DataFrame.from_dict(data_table)

        # Compute door count and location based on the selected door number
        DoorCount_df = compute_door_count_and_location(df1, threshold_time='03:00:00', ensure_non_negative=True, door_number=door_number).dropna()

        # Generate the line plot for the selected door
        fig_door_flow = plot_passenger_door_flow(DoorCount_df)

        return fig_door_flow

    # Return an empty figure if no data is available
    return go.Figure()


# Callback to generate the trajectory plot
@app.callback(
    Output('trajectory-map', 'figure'),
    [Input('stored-data', 'data')],
    prevent_initial_call=True
)
def update_trajectory_map(stored_data):
    if stored_data:
        # Reconstruct the location_df from the stored data
        result_df = pd.DataFrame.from_dict(stored_data)

        # Ensure 'datetime' column is in datetime format
        result_df['datetime'] = pd.to_datetime(result_df['datetime'], errors='coerce')

        # Call the function to generate the animated map
        fig = create_animated_trajectory(result_df)

        return fig

    # Return an empty figure if no data
    return go.Figure()

@app.callback(
    Output('scatter-map', 'figure'),
    [Input('filter-time-btn', 'n_clicks')],
    [State('start-time-input', 'value'),
     State('end-time-input', 'value'),
     State('data-table', 'data')],
    prevent_initial_call=True
)
def update_scatter_map(n_clicks, start_time, end_time, data_table):
    if data_table:
        # Convert the 'data-table' content back into a DataFrame
        df1 = pd.DataFrame.from_dict(data_table)

        # Compute result_df from df1 instead of read from loaded data to make sure no negative entries with ensure_non_negative=True
        result_df = compute_passenger_count_and_location(df1, threshold_time='03:00:00', ensure_non_negative=True)

        Station_GPS = pd.read_csv('data/Station_GPS.csv')
        result_df = map_nearest_station(result_df.dropna(), Station_GPS)

        result_df['datetime'] = pd.to_datetime(result_df['datetime'], errors='coerce')

        # Extract the date from the first row's 'datetime' column
        date = result_df['datetime'].dt.date.iloc[0]

        # Combine the extracted date with the provided times
        start_time_full = pd.to_datetime(f"{date} {start_time}", errors='coerce')
        end_time_full = pd.to_datetime(f"{date} {end_time}", errors='coerce')

        # Filter the dataframe based on the time slot
        filtered_result = result_df[(result_df['datetime'] >= start_time_full) & (result_df['datetime'] <= end_time_full)]

        # Define the color category based on the number of passengers on train
        filtered_result['passenger_number'] = np.where(
            filtered_result['passengers_on_train'] > 70, 'More than 70',
            np.where(filtered_result['passengers_on_train'].between(20, 70), 'Between 20 and 70', 'Less than 20')
        )

        # Plot the scatter map, using the 'passenger_category' column for color
        fig = px.scatter_mapbox(filtered_result, lat="latitude", lon="longitude",
                                color="passenger_number",
                                color_discrete_map={'More than 70': 'red', 'Between 20 and 70': 'blue',
                                                    'Less than 20': 'green'},  # Map categories to actual colors
                                size="passengers_on_train", size_max=15, zoom=11,
                                hover_name="Station_name",  # Ensure station name appears in hover info
                                hover_data={'latitude': False, 'longitude': False, 'passenger_number': False,
                                            'datetime': True})  # Hide these in hover

        # Customize the map's appearance and set mapbox style
        fig.update_layout(mapbox_style="open-street-map")

        return fig

    return go.Figure()

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
    [Input('stored-data', 'data')],
    prevent_initial_call=True
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

# Callback to update the pie charts based on user input threshold
@app.callback(
    [Output('pie-in-chart', 'figure'),
     Output('pie-out-chart', 'figure')],
    [Input('stored-data', 'data'),
     Input('pie-threshold-input', 'value')],
    prevent_initial_call=True
)
def update_pie_charts(stored_data, pie_threshold):
    if stored_data and pie_threshold is not None:
        # Reconstruct the result_df from the stored data
        result_df = pd.DataFrame.from_dict(stored_data)

        # Group and aggregate data for pie charts
        pie_in_df = result_df.groupby(by="Station_name", dropna=False).agg({'APC_Count_In': 'sum'}).reset_index()
        pie_in_df.loc[pie_in_df['APC_Count_In'] < pie_threshold, 'Station_name'] = 'Other stations'

        pie_out_df = result_df.groupby(by="Station_name", dropna=False).agg({'APC_Count_Out': 'sum'}).reset_index()
        pie_out_df.loc[pie_out_df['APC_Count_Out'] < pie_threshold, 'Station_name'] = 'Other stations'

        # Create pie charts
        fig_in = px.pie(pie_in_df, values='APC_Count_In', names='Station_name', title='Daily APC Count In by Station')
        fig_out = px.pie(pie_out_df, values='APC_Count_Out', names='Station_name', title='Daily APC Count Out by Station')

        return fig_in, fig_out

    # Return empty figures if no data is available
    return go.Figure(), go.Figure()


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
