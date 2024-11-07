import plotly.express as px
import pandas as pd

def create_animated_trajectory(location_df):
    """
    Creates an animated plot showing the vehicle's location over time on a map.
    The map will automatically zoom and center based on the location data provided.
    """
    # Ensure 'datetime' column is of datetime type for proper animation
    location_df['datetime'] = pd.to_datetime(location_df['datetime'])

    # Sort the dataframe by datetime to ensure proper animation
    location_df = location_df.sort_values(by='datetime')

    # Create an animated scatter plot using Plotly Express on a Mapbox map
    fig = px.scatter_mapbox(
        location_df,
        lat='latitude',
        lon='longitude',
        animation_frame='datetime',
        mapbox_style="open-street-map",
        zoom=10,  # Default zoom level
        title="Vehicle Trajectory over Time",
        hover_name='datetime',
        hover_data={'latitude': True, 'longitude': True,'Station_name': True}
    )

    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    # Show the figure
    return fig


def plot_passenger_door_flow(df):
    """
    Creates a line plot of 'datetime' against 'APC_Count_In' and 'APC_Count_Out'.

    Parameters:
    df (pd.DataFrame): The DataFrame containing 'datetime', 'APC_Count_In', and 'APC_Count_Out'.

    Returns:
    fig: A plotly figure containing the line plot.
    """
    # Ensure 'datetime' column is in datetime format (in case it is not)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Melt the DataFrame so that 'APC_Count_In' and 'APC_Count_Out' are plotted as separate lines
    df_melted = df.melt(id_vars='datetime', value_vars=['APC_Count_In', 'APC_Count_Out'],
                        var_name='Passenger Flow', value_name='Count')

    # Create the line plot
    fig = px.line(
        df_melted,
        x='datetime',
        y='Count',
        color='Passenger Flow',  # This will distinguish between APC_Count_In and APC_Count_Out
        title='Passenger Flow Over Time (Door-wise)',
        labels={
            'datetime': 'Date and Time',
            'Count': 'Passenger Count',
            'Passenger Flow': 'Type'
        }
    )

    # Customize layout if necessary
    fig.update_layout(
        xaxis_title='Date and Time',
        yaxis_title='Passenger Count',
        legend_title='Passenger Flow at Door'
    )

    return fig