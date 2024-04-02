import click
from loguru import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def setup_logger(log_to_file: bool, log_level: str) -> None:
    """
    Configure the logger based on user preferences.

    Args:
    log_to_file (bool): If True, logs are written to a file. If False, logs are printed to the console.
    log_level (str): Level of logging. Example: 'INFO', 'ERROR'.

    Returns:
    None
    """
    if log_to_file:
        logger.add("debug.log", format="{time} {level} {message}", level=log_level.upper())
    else:
        logger.remove()
        logger.add(lambda msg: click.echo(msg, err=True), format="{time} {level} {message}", level=log_level.upper())


def generate_time_series(num_registers: int, gap_percentage: float, trend: str, seasonality: float, seed: int) -> pd.DataFrame:
    """
    Generates a time series dataset with optional trend and seasonality.

    Args:
    num_registers (int): Number of data points in the series.
    gap_percentage (float): Percentage of gaps to introduce in the data.
    trend (str): Type of trend to add ('none', 'linear', 'exponential').
    seasonality (float): Frequency of seasonality.
    seed (int): Seed for random number generation.

    Returns:
    pd.DataFrame: DataFrame containing timestamps and data values.
    """

    if seed is not None:
        np.random.seed(seed)
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i) for i in range(num_registers)]

    if trend == 'linear':
        values = np.linspace(0, 1, num_registers)
    elif trend == 'exponential':
        values = np.exp(np.linspace(0, 1, num_registers))
    else:
        values = np.random.rand(num_registers)

    if seasonality:
        seasonal_effect = np.sin(np.linspace(0, seasonality * np.pi, num_registers))
        values *= seasonal_effect

    if 0 < gap_percentage < 100:
        gap_indices = np.random.choice(range(num_registers), int(num_registers * gap_percentage / 100), replace=False)
        values[gap_indices] = np.nan

    return pd.DataFrame({'Timestamp': timestamps, 'Value': values})


def apply_smoothing(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Applies simple moving average smoothing to the data.

    Args:
    data (pd.DataFrame): DataFrame of the time series.
    window_size (int): Window size for smoothing.

    Returns:
    pd.DataFrame: DataFrame with an additional 'Smoothed' column for the smoothed data.
    """
    data['Smoothed'] = data['Value'].rolling(window=window_size, min_periods=1).mean()
    return data


def save_plot(data: pd.DataFrame, file_name: str) -> None:
    """
    Saves a plot of the time series data.

    Args:
    data (pd.DataFrame): DataFrame of the time series.
    file_name (str): File name to save the plot.

    Returns:
    None: This function does not return any value, but saves an image of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data['Timestamp'], data['Value'], label='Original')
    if 'Smoothed' in data.columns:
        plt.plot(data['Timestamp'], data['Smoothed'], label='Smoothed', alpha=0.7)
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.title('Time Series Data')
    plt.legend()
    plt.savefig(file_name)
    plt.close()

@click.command()
@click.argument('num_registers', type=int)
@click.option('--gap-percentage', default=0, type=float, help="Percentage of gaps to introduce.")
@click.option('--trend', type=click.Choice(['none', 'linear', 'exponential'], case_sensitive=False), default='none', help="Type of trend to add.")
@click.option('--seasonality', default=0, type=float, help="Frequency of seasonality.")
@click.option('--smoothing', default=10, type=int, help="Window size for data smoothing.")
@click.option('--plot-file', default='time_series_plot.png', type=str, help="File name for saving the plot.")
@click.option('--seed', default=None, type=int, help="Seed for random number generation.")
@click.option('--log-to-file', is_flag=True, help="Enable logging to a file.")
@click.option('--log-level', default='INFO', type=str, help="Set the logging level.")
def main(num_registers: int, gap_percentage: float, trend: str, seasonality: float, smoothing: int, plot_file: str, log_to_file: bool, log_level: str, seed: int) -> None:
    """
    CLI tool to generate and visualize a time series data set with optional trend, seasonality, and smoothing.

    The tool allows generating a time series with a specified number of data points (num_registers). 
    It provides options to introduce gaps in data, add a trend (none, linear, exponential), 
    incorporate seasonality, apply data smoothing, and save a plot of the series.

    Args:
    num_registers (int): Number of data points in the series.
    gap_percentage (float): Percentage of gaps to introduce in the data.
    trend (str): Type of trend to add ('none', 'linear', 'exponential').
    seasonality (float): Frequency of seasonality.
    smoothing (int): Window size for data smoothing.
    plot_file (str): File name for saving the plot.
    log_to_file (bool): Enable logging to a file.
    log_level (str): Logging level (e.g., 'INFO', 'ERROR').

    Returns:
    None: This function does not return any value, but generates a CSV file and a plot image.
    """
    setup_logger(log_to_file, log_level)

    try:
        df = generate_time_series(num_registers, gap_percentage, trend, seasonality, seed)
        if smoothing > 0:
            df = apply_smoothing(df, smoothing)
        df.to_csv('time_series.csv', index=False)
        save_plot(df, plot_file)
        logger.info("Time series data generated and plot saved.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()



