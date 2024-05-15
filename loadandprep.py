
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
from tensorflow.keras.callbacks import ModelCheckpoint


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
    # print(f"redadass--------------------- {filename}")
    with np.load(filename) as data:
        # print(f"Data has been successfully loaded from {filename}")
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
    
def dataset_generator(file_paths, batch_size, window_size, augmentations=[]):
    
    all_timestamps = []
    all_values = []
    
    # The LSTM input layer must be 3D, dimensions are: samples, time steps, and features.
    # Assuming read_data_from_npz returns lists
    for file in file_paths:
        timestamps, values = read_data_from_npz(file)
        all_timestamps.extend(timestamps)
        all_values.extend(values)

    tuple_array = np.array(list(zip(all_timestamps, all_values)))
    
    train_size = int(len(tuple_array) * 0.8)
    train_data, test_data = tuple_array[:train_size], tuple_array[train_size:]
    
    trainX, trainY = [], []
    # train Y dimension = window_size
    for i in range(len(train_data) - batch_size):
        trainX.append(train_data[i:i+batch_size, 1])
        trainY.append(train_data[i+batch_size:i+batch_size+window_size, 1])
        
    testX, testY = [], []
    # test Y dimension = window_size
    for i in range(len(test_data) - batch_size):
        testX.append(test_data[i:i+batch_size, 1])
        testY.append(test_data[i+batch_size:i+batch_size+window_size, 1])
                
    print (f"TrainX last: {trainX[len(trainX)-1]}")
    print (f"TrainY last: {trainY[len(trainY)-1]}")
    print (f"TestX last: {testX[len(testX)-1]}")
    print (f"TestY last: {testY[len(testY)-1]}")
    
    contador = 0
    for i in range(len(trainX)):
        if len(trainX[i]) != batch_size:
            contador += 1
    
    trainX = trainX[:len(trainX)-contador]
    trainY = trainY[:len(trainY)-contador]
    
    contador = 0
    for i in range(len(trainY)):
        if len(trainY[i]) != window_size:
            contador += 1
    
    trainX = trainX[:len(trainX)-contador]
    trainY = trainY[:len(trainY)-contador]
    
    contador = 0
    for i in range(len(testX)):
        if len(testX[i]) != batch_size:
            contador += 1
    
    testX = testX[:len(testX)-contador]
    testY = testY[:len(testY)-contador]

    contador = 0
    for i in range(len(testY)):
        if len(testY[i]) != window_size:
            contador += 1
    
    testX = testX[:len(testX)-contador]
    testY = testY[:len(testY)-contador]
    
        
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)
    
    # trainX = augmentation_operations(trainX, augmentations)
    # trainY = augmentation_operations(trainY, augmentations)
    # testX = augmentation_operations(testX, augmentations)
    # testY = augmentation_operations(testY, augmentations)
    
    return trainX, trainY, testX, testY

    
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

        # augmentations = ['normalize', 'add_noise', 'smooth', 'remove_outliers', 'remove_nans', 'remove_duplicates', 'magnitude_warping', 'scaling', 'time_warping']
        augmentations = ['add_noise']
        
        batch_size = 32
        window_size = 5
        trainX, trainY, testX, testY = dataset_generator(file_paths, batch_size, window_size, augmentations)
        
        print("------------", trainX[0])
        print("................" , trainY[0])
        
        print (f"TrainX shape: {trainX.shape}")
        print (f"TrainY shape: {trainY.shape}")
        print (f"TestX shape: {testX.shape}")
        print (f"TestY shape: {testY.shape}")
        
        # data_train = tf.data.Dataset.from_tensor_slices((trainX, trainY))
        # data_test = tf.data.Dataset.from_tensor_slices((testX, testY))        
                
        # Two models are created one convolutional and one recurrent (LSTM)
        # The models are trained with the training data and evaluated with the test data
        # The model with the best performance is selected
        # The model is saved
        # The model is used to predict the next value of the time series
        
        # conv_model = tf.keras.Sequential([
        #     tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        #     tf.keras.layers.Dense(units=32, activation='relu'),
        #     tf.keras.layers.Dense(units=1)
        # ])
        
        # conv_model.compile(optimizer='adam', loss='mse')
        # conv_model.fit(data_train, epochs=10)
                
        # conv_loss = conv_model.evaluate(data_test)
        # print(f"Convolutional model loss: {conv_loss}")
        
        model_version = 1
        dataset_version = 1
        
        lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(batch_size, 1)),
            tf.keras.layers.LSTM(units=50),
            tf.keras.layers.Dense(units=8),
            tf.keras.layers.Dense(units=window_size)
        ])
        
        lstm_model.summary()
        
        metrics=[tf.metrics.MeanAbsoluteError()]
        
        lstm_model.compile(optimizer='adam', loss='mse', metrics=metrics)
        # needed shape -> samples, time steps, and features
        
        checkpoint_filepath = f'./weights/model_{model_version}_dataset_{dataset_version}_{{loss:.3f}}.weights.h5'
        
        checkpoint_callback = ModelCheckpoint(
            checkpoint_filepath,
            monitor='loss',            
            mode='min',                    
            save_weights_only=True,        
            save_best_only=True,          
            verbose=1                      
        )

        lstm_model.fit(trainX, trainY, epochs=3, batch_size=32, callbacks=[checkpoint_callback])
        
        # history = model.fit(
        #     x=train_generator,
        #     steps_per_epoch=len(train_features) // batch_size,
        #     epochs=80,
        #     validation_data=test_generator,
        #     validation_steps=len(test_features) // b
                
        # lstm_model.fit(data_train, epochs=10)
        
        # plt.plot(lstm_model.history.history['loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.show()
        
        # trainScore = lstm_model.evaluate(trainX, trainY, verbose=0)
        # print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))
        # testScore = lstm_model.evaluate(testX, testY, verbose=0)
        # print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))
        
        prediction = lstm_model.predict(testX)
        print(f"prediction 0 :", prediction[0])
        print(f"testX 0 :", testX[0])
        print(f"testY 0 :", testY[0])
        # print(f"Prediction shape: {prediction.shape}")
        # que la entrada se vea en otro color (verde)
        plt.plot(testX, color='green')
        
        plt.plot( testY, color='red')
        # que la predicción se ve al final de la serie temporal en otro color (azul)
        plt.plot(prediction, color='blue')
        plt.legend(['Input', 'Real', 'Prediction'])
        plt.title('Model prediction')
        plt.ylabel('Value')
        plt.xlabel('Index')
        plt.show()
    
    elif operation == 'load_and_train_model':        
        # model.load_weights(best_filepath)
        pass
    
    else:
        print("Invalid operation. Please choose one of the following: 'stacionary_and_correlation', 'npz_creation', 'load_and_generate_data'")
        
    
    return None


if __name__ == '__main__':
    main()
