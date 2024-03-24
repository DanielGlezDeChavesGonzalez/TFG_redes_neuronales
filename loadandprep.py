
import psycopg2
import numpy as np
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, pperron
import click
from loguru import logger
from typing import Any, Callable
from matplotlib import pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
from scipy import stats

# Database connection parameters
dbname = "datos_temporales"
host = "postgres"
port = "5432"
user = "postgres"
password = "postgres"

def load_data_from_database ():
    # Connect to your postgres DB
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
    
    # Open a cursor to perform database operations
    cur = conn.cursor()
    
    # Get all data from all tables in the database
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    
    tables = cur.fetchall()
    table_rows = []
    for table in tables:
        cur.execute(f"SELECT * FROM {table[0]}")
        table_rows.append(cur.fetchall())

    # Close the cursor and connection
    cur.close()
    conn.close()
    
    return table_rows


# augmented dickey fuller test represented with graphs with plt
def adf_test(data: pd.Series) -> None:
    result = adfuller(data)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    return None

# Kwiatkowski-Phillips-Schmidt-Shin test represented with graphs with plt
def kpss_test(data: pd.Series) -> None:
    result = kpss(data)
    print(f'KPSS Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[3].items():
        print(f'\t{key}: {value}')
    return None

# Autocorrelation and Partial Autocorrelation Function represented with graphs with plt
def acf_pacf(data: pd.Series) -> None:
    acf_vals = acf(data)
    pacf_vals = pacf(data)
    print('ACF:')
    print(acf_vals)
    print('PACF:')
    print(pacf_vals)
    return None

# Phillips-Perron test represented with graphs with plt
def pp_test(data: pd.Series) -> None:
    result = pperron(data)
    print(f'PP Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    return None


def slice_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

def create_npz(data, filename):
    timestamps = [x[0].timestamp() for x in data]
    values = [x[1] for x in data]
    np.savez(filename, timestamps=timestamps, values=values)
    
def read_data_from_npz(filename):
    with np.load(filename) as data:
        return data['timestamps'], data['values']
    
def augmentation_operations(data: pd.Series, operation: str) -> pd.Series:
    if operation == 'normalize':
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif operation == 'add_noise':
        noise = np.random.normal(loc=0.0, scale=0.05, size=data.shape)
        data = data + noise
    return data
    
def data_generator_npz(file_paths, batch_size, augmentations=[]):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(read_data_from_npz(x)))
    
    # Apply data augmentation
    dataset = dataset.map(lambda x, y: augmentation_operations(x, y, augmentations))
    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    return dataset

    
@click.command()
@click.argument('type_operation', type=str , default='load_data_from_database', 
                help="Type of operation to perform. Options: stacionary_and_correlation, npz_creation, load_and_generate_data")
@click.option('--folder', type=str, default='.', help="Folder where the data is stored.")

def main(operation: str , folder : str) -> None:
    
    data = load_data_from_database()

    if operation == 'stacionary_and_correlation':
        adf_test(data)
        kpss_test(data)
        acf_pacf(data)
        pp_test(data)
    elif operation == 'npz_creation':
        chunk_size = 10000
        sliced_data = slice_data(data, chunk_size)
        for idx, chunk in enumerate(sliced_data):
            npz_filename = f'data_chunk_{idx}.npz'
            create_npz(chunk, npz_filename)
        print("Data has been successfully saved in NPZ format.")
    elif operation == 'load_and_generate_data':
        file_paths = ['data_chunk_0.npz', 'data_chunk_1.npz']
        batch_size = 64
        augmentations = ['normalize', 'add_noise']
        dataset = data_generator_npz(file_paths, batch_size, augmentations)
        print("Data has been successfully loaded and augmented.")
        plt.plot(dataset)
    else:
        print("Invalid operation. Please choose one of the following: stacionary_and_correlation, npz_creation")
        
    
    return None


if __name__ == '__main__':
    main()
