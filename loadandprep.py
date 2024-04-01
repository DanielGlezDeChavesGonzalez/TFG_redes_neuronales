
import psycopg2
import numpy as np
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch.unitroot import PhillipsPerron
import click
from loguru import logger
from typing import Any, Callable
from matplotlib import pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
from scipy import stats
import os

# Database connection parameters
dbname = "datos_temporales"
host = "postgres"
port = "5432"
user = "postgres"
password = "postgres"

def load_data_from_folder(folder: str) -> Any:
    data = []
    for file in os.listdir(folder):
        # Lee el archivo CSV y asigna los nombres de las columnas
        df = pd.read_csv(os.path.join(folder, file), sep=';', names=['Timestamp', 'Value'])
        data.append(df)
        print(f"Data from file {file}")
        print(data[-1].head())
        # Gráfico de la serie temporal
        
        plt.plot(data[-1]['Value'])
        plt.title('Serie temporal')
        plt.xlabel('Índice')
        plt.ylabel('Valor')
        plt.show()
        
        
    return data

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
def adf_test(data: pd.Series, folder: str) -> None:
    print("ADF Test")
    for df, file in zip(data, os.listdir(folder)):
        values = df['Value']
        
        # Realizar la prueba de Dickey-Fuller en la columna 'Value'
        result = adfuller(values)
        
        # Imprimir los resultados
        print('DataFrame: ' + file)
        print(df.head())
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')

# Kwiatkowski-Phillips-Schmidt-Shin test represented with graphs with plt
def kpss_test(data: pd.Series, folder : str) -> None:
    print("KPSS Test")
    for df, file in zip(data, os.listdir(folder)):
        values = df['Value']
        
        # Realizar la prueba de KPSS en la columna 'Value'
        result = kpss(values)
        
        # Imprimir los resultados
        print('DataFrame: ' + file)
        print(df.head())
        print(f'KPSS Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[3].items():
            print(f'\t{key}: {value}')
        

# Autocorrelation and Partial Autocorrelation Function represented with graphs with plt
def acf_pacf(data: pd.Series, folder : str) -> None:
    print("ACF and PACF")
    for df, file in zip(data, os.listdir(folder)):
        values = df['Value']
        
         # Crear una figura con un panel dividido en 1 fila y 2 columnas
        plt.figure(figsize=(12, 5))
        
        # Gráfico de la función de autocorrelación
        plt.subplot(2, 1, 1)
        plot_acf(values, ax=plt.gca())
        plt.title(f'ACF - {file}')
        
        # Gráfico de la función de autocorrelación parcial
        plt.subplot(2, 1, 2)
        plot_pacf(values, ax=plt.gca())
        plt.title(f'PACF - {file}')
        
        # Ajustar los gráficos
        plt.subplots_adjust(hspace=0.5)
        
        # Mostrar los gráficos
        plt.show()


# Phillips-Perron test represented with graphs with plt
def pp_test(data: pd.Series, folder : str) -> None:
    print("PP Test")
    for df, file in zip(data, os.listdir(folder)):
        values = df['Value']
        
        # Realizar la prueba de Phillips-Perron en la columna 'Value'
        result = PhillipsPerron(values)
        
        # Imprimir los resultados
        print('DataFrame: ' + file)
        print(f'PP Statistic: {result.stat}')
        print(f'p-value: {result.pvalue}')
        print(f'Critical Values: {result.critical_values}')
        print(f'Null Hypothesis: {result.null_hypothesis}')
        print(f'Alternative Hypothesis: {result.alternative_hypothesis}')

            
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
        return data['Timestamp'], data['Value']
    
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
@click.argument('operation', type=str , default='stacionary_and_correlation')
@click.option('--folder', type=str, default='.', help="Folder where the data is stored.")

def main(operation: str , folder : str) -> None:
    
    if folder:
        logger.info(f"Data will be loaded from {folder}")
        data = load_data_from_folder(folder)
    else:
        logger.info(f"Data will be loaded from the database")
        data = load_data_from_database()

    if operation == 'stacionary_and_correlation':
        adf_test(data, folder)
        kpss_test(data, folder)
        acf_pacf(data, folder)
        pp_test(data, folder)
        
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
