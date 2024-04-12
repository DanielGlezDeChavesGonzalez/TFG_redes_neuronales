
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
import IPython
import IPython.display

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
        # print(data[-1].head())
        # Gráfico de la serie temporal
        
        # plt.plot(data[-1]['Value'])
        # plt.title('Serie temporal')
        # plt.xlabel('Índice')
        # plt.ylabel('Valor')
        # plt.show()
        
        
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
def adf_test(data: pd.Series) -> None:
    print("ADF Test-----------------------------------------")
    values = data['Value']
    
    # Realizar la prueba de Dickey-Fuller en la columna 'Value'
    result = adfuller(values)
    
    # Imprimir los resultados
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Kwiatkowski-Phillips-Schmidt-Shin test represented with graphs with plt
def kpss_test(data: pd.Series) -> None:
    print("KPSS Test-----------------------------------------")
    values = data['Value']
    
    # Realizar la prueba de KPSS en la columna 'Value'
    result = kpss(values)
    
    # Imprimir los resultados
    print(f'KPSS Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[3].items():
        print(f'\t{key}: {value}')
        

# Autocorrelation and Partial Autocorrelation Function represented with graphs with plt
def acf_pacf(data: pd.Series) -> None:
    print("ACF and PACF-----------------------------------------")
    values = data['Value']
    # Crear una figura con un panel dividido en 1 fila y 2 columnas
    plt.figure(figsize=(12, 5))
    # Gráfico de la función de autocorrelación
    plt.subplot(2, 1, 1)
    plot_acf(values, ax=plt.gca())
    plt.title(f'ACF')
    # Gráfico de la función de autocorrelación parcial
    plt.subplot(2, 1, 2)
    plot_pacf(values, ax=plt.gca())
    plt.title(f'PACF')
    # Ajustar los gráficos
    plt.subplots_adjust(hspace=0.5)
    # Mostrar los gráficos
    plt.show()


# Phillips-Perron test represented with graphs with plt
def pp_test(data: pd.Series) -> None:
    print("PP Test-----------------------------------------")
    values = data['Value']
    
    # Realizar la prueba de Phillips-Perron en la columna 'Value'
    result = PhillipsPerron(values)
    
    # Imprimir los resultados
    print(f'PP Statistic: {result.stat}')
    print(f'p-value: {result.pvalue}')
    print(f'Critical Values: {result.critical_values}')
    print(f'Null Hypothesis: {result.null_hypothesis}')
    print(f'Alternative Hypothesis: {result.alternative_hypothesis}')

            
    return None

def slice_data(data, chunk_size):
    print (f"Data will be sliced in chunks of {chunk_size}")
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]
        print(f"Data chunk {i} to {i+chunk_size} has been sliced.")
        
def create_npz(data, filename, folder_save ):
    print(f"Data will be saved in {filename}")
    timestamps = data['Timestamp']
    values = data['Value']
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    np.savez(os.path.join(folder_save, filename), Timestamp=timestamps, Value=values)
    print(f"Data has been successfully saved in {filename}")
    
def read_data_from_npz(filename):
    with np.load(filename) as data:
        return data['Timestamp'], data['Value']
    
def augmentation_operations(data, augmentations):
    if 'normalize' in augmentations:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    if 'add_noise' in augmentations:
        noise = np.random.normal(loc=0.0, scale=0.05, size=data.shape)
        data = data + noise
    if 'smooth' in augmentations:
        data = pd.Series(data).rolling(window=5, min_periods=1).mean()
    if 'remove_outliers' in augmentations:
        z_scores = np.abs(stats.zscore(data))
        data = data[(z_scores < 3)]
    if 'remove_nans' in augmentations:
        data = data[~np.isnan(data)]
    if 'remove_duplicates' in augmentations:
        data = data.drop_duplicates()
    if 'magnitude_warping' in augmentations:
        data = data * np.random.uniform(0.9, 1.1)
    if 'scaling' in augmentations:
        data = data * np.random.uniform(0.5, 2.0)
    if 'time_warping' in augmentations:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data = data[indices]
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
@click.option('--folder-read', type=str, default='.', help="Folder where the data is stored.")
@click.option('--folder-save', type=str, default='.', help="Folder where the data will be saved.")

def main(operation: str , folder_read : str, folder_save: str) -> None:
    
    if operation != 'load_and_generate_data':
        if folder_read:
            # python .\loadandprep.py operation --folder-read .\datos_sensores_prueba\
            logger.info(f"Data will be loaded from {folder_read}")
            data = load_data_from_folder(folder_read)
        else:
            # python .\loadandprep.py stacionary_and_correlation
            logger.info(f"Data will be loaded from the database")
            data = load_data_from_database()

    if operation == 'stacionary_and_correlation':
        # python .\loadandprep.py stacionary_and_correlation --folder-read .\datos_sensores\
        for df, file in zip(data, os.listdir(folder_read)):
            print(f"Data analize from file {file}")
            # chunks = slice_data(df, 10000)
            # for chunk in chunks:
            #     adf_test(chunk)
            #     kpss_test(chunk)
            #     # acf_pacf(chunk)
            #     pp_test(chunk)
            adf_test(df)
            kpss_test(df)
            # acf_pacf(df)
            pp_test(df)
        
    elif operation == 'npz_creation':
        # python .\loadandprep.py npz_creation --folder-read .\datos_sensores_prueba\ --folder-save datos_npz
        chunk_size = 10000
        unified_data = pd.concat(data)
        # sliced_data = slice_data(data, chunk_size)
        sliced_data = slice_data(unified_data, chunk_size)
        for idx, chunk in enumerate(sliced_data):
            npz_filename = f'data_chunk_{idx}.npz'
            create_npz(chunk, npz_filename, folder_save)
        print("Data has been successfully saved in NPZ format.")
        
    elif operation == 'load_and_generate_data':
        # python .\loadandprep.py load_and_generate_data --folder-read datos_npz
        file_paths = [os.path.join(folder_read, f) for f in os.listdir(folder_read)]
        batch_size = 64
        # augmentations = ['normalize', 'add_noise', 'smooth', 'remove_outliers', 'remove_nans', 'remove_duplicates', 'magnitude_warping', 'scaling', 'time_warping']
        augmentations = ['add_noise']
        print("Data will be loaded and generated.")
        dataset = data_generator_npz(file_paths, batch_size, augmentations)
        for data in dataset:
            print(data.head())
            
        print("Data has been loaded and generated.")
        conv_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        conv_model.compile_and_fit(dataset)
        
        plot = tf.keras.utils.plot_model(conv_model, show_shapes=True)
        
        IPython.display.Image(plot)
        
            
        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dense(1)
        ])
        # print('Input shape:', wide_window.example[0].shape)
        # print('Output shape:', lstm_model(wide_window.example[0]).shape)
        
        lstm_model.compile_and_fit(dataset)
        
        plot = tf.keras.utils.plot_model(lstm_model, show_shapes=True)
        
        IPython.display.clear_output()
        
        IPython.display.Image(plot)
        
        # compara el rendimiento de los modelos
        
        # val_performance = {}
        # performance = {}
        # val_performance['Conv'] = conv_model.evaluate(dataset)
        # performance['Conv'] = conv_model.evaluate(dataset)
        
        # val_performance['LSTM'] = lstm_model.evaluate(dataset)
        # performance['LSTM'] = lstm_model.evaluate(dataset)
        
        # IPython.display.clear_output()
        
        # for name, value in performance.items():
        #     print(f'{name:12s}: {value[1]:0.4f}')
        # print()
        
    

        
    else:
        print("Invalid operation. Please choose one of the following: 'stacionary_and_correlation', 'npz_creation', 'load_and_generate_data'")
        
    
    return None


if __name__ == '__main__':
    main()
