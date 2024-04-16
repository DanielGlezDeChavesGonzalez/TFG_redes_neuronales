
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
        
    print(f"redadass--------------------- {filename}")
    with np.load(filename) as data:
        return data['timestamps'], data['values']
    
     
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
    dataset = dataset.flat_map(lambda filename: tf.data.Dataset.from_tensor_slices(read_data_from_npz(filename)))
    
    # Apply data augmentation
    dataset = dataset.map(lambda timestamps, values: (timestamps, augmentation_operations(values, augmentations)))

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
        # print (file_paths)
        batch_size = 64
        # augmentations = ['normalize', 'add_noise', 'smooth', 'remove_outliers', 'remove_nans', 'remove_duplicates', 'magnitude_warping', 'scaling', 'time_warping']
        augmentations = ['add_noise']
        print("Data will be loaded and generated.")
        dataset = data_generator_npz(file_paths, batch_size, augmentations)
        # for data in dataset:
        #     print(data.head())
        
        print("Data has been loaded and generated.")
                
        # data_train is the first 85% of the data
        data_train = dataset.take(int(0.85 * len(dataset)))
        
        # data_test is the last 15% of the data    
        data_test = dataset.skip(int(0.85 * len(dataset)))
        
        # Two models are created one convolutional and one recurrent (LSTM)
        # The models are trained with the training data and evaluated with the test data
        # The model with the best performance is selected
        # The model is saved
        # The model is used to predict the next value of the time series
        
        # conv_model = tf.keras.Sequential([
        #     tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        #     tf.keras.layers.MaxPooling1D(pool_size=2),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(50, activation='relu'),
        #     tf.keras.layers.Dense(1)
        # ])
        conv_model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])
        
        lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(1)
        ])
        
        conv_model.compile(optimizer='adam', loss='mse')
        lstm_model.compile(optimizer='adam', loss='mse')
        
        conv_model.fit(data_train, epochs=10)
        lstm_model.fit(data_train, epochs=10)
        
        conv_loss = conv_model.evaluate(data_test)
        lstm_loss = lstm_model.evaluate(data_test)
        print(f"Convolutional model loss: {conv_loss}")
        print(f"LSTM model loss: {lstm_loss}")
        
        # Plot predictions of the models in same window 
        plt.figure(figsize=(12, 6))
        plt.plot(data_test, label='True data')
        plt.plot(conv_model.predict(data_test), label='Convolutional model')
        plt.plot(lstm_model.predict(data_test), label='LSTM model')
        plt.legend()
        plt.show()
        
        if conv_loss < lstm_loss:
            model = conv_model
            
        print("Model " + model + " has been selected.")
        
        model.save('model.h5')
        
        # Predict the next value of the time series
        next_value = model.predict(data_test)
        print(f"Next value of the time series: {next_value}")
        
        
    else:
        print("Invalid operation. Please choose one of the following: 'stacionary_and_correlation', 'npz_creation', 'load_and_generate_data'")
        
    
    return None


if __name__ == '__main__':
    main()
