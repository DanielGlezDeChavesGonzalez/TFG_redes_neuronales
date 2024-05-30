
import numpy as np
import tensorflow as tf
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
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import dask.dataframe as dd
from tensorflow.keras.models import Sequential # type: ignore
from model_creators import Lstm_model, Conv1D_model, Dense_model

def augmentation_operations(data, augmentations):
    augmented_data = data.copy()
    for augmentation in augmentations:
        if augmentation == 'normalize':
            augmented_data = (augmented_data - augmented_data.mean()) / augmented_data.std()
        elif augmentation == 'add_noise':
            augmented_data = augmented_data + np.random.normal(0, 0.1, augmented_data.shape)
        elif augmentation == 'smooth':
            augmented_data = augmented_data.rolling(window=5).mean()
        elif augmentation == 'remove_outliers':
            augmented_data = augmented_data[(np.abs(stats.zscore(augmented_data)) < 3).all(axis=1)]
        elif augmentation == 'remove_nans':
            augmented_data = augmented_data.dropna()
        elif augmentation == 'remove_duplicates':
            augmented_data = augmented_data.drop_duplicates()
        elif augmentation == 'magnitude_warping':
            augmented_data = augmented_data * np.random.uniform(0.9, 1.1)
        elif augmentation == 'scaling':
            augmented_data = augmented_data * np.random.uniform(0.5, 1.5)
        elif augmentation == 'time_warping':
            augmented_data = augmented_data.sample(frac=1).reset_index(drop=True)
            
    return augmented_data
    
def read_data_from_npz(filename):
    # print(f"Reading data from {filename}")
    with np.load(filename) as data:
        timestamps = data['Timestamp']
        values = data['Value']
        return timestamps, values

def read_and_combine_data(file_paths):
    all_timestamps = []
    all_values = []

    for file in file_paths:
        timestamps, values = read_data_from_npz(file)
        all_timestamps.extend(timestamps)
        all_values.extend(values)
        
    return np.array(list(zip(all_timestamps, all_values)))

def split_data(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    return data[:train_size], data[train_size:]

def create_sequences(data, batch_size, window_size):
    X, Y = [], []
    for i in range(len(data) - batch_size):
        X.append(data[i:i + batch_size, 1])
        Y.append(data[i + batch_size:i + batch_size + window_size, 1])
    return X, Y

def adjust_batch_sizes(X, Y, expected_size_X, expected_size_Y):
    adjusted_X = []
    adjusted_Y = []

    for x, y in zip(X, Y):
        if len(x) == expected_size_X and len(y) == expected_size_Y:
            adjusted_X.append(x)
            adjusted_Y.append(y)

    return adjusted_X, adjusted_Y

def dataset_generator(file_paths, batch_size, window_size, augmentations=[], train_ratio=0.8):
    combined_data = read_and_combine_data(file_paths)
    
    print (f"Data has been read and combined")
    train_data, test_data = split_data(combined_data, train_ratio)
    
    print (f"Data has been splitted")
    trainX, trainY = create_sequences(train_data, batch_size, window_size)
    testX, testY = create_sequences(test_data, batch_size, window_size)
    
    print (f"Sequences have been created")
    trainX, trainY = adjust_batch_sizes(trainX, trainY, batch_size, window_size)
    testX, testY = adjust_batch_sizes(testX, testY, batch_size, window_size)
    
    print (f"Splitted data, created sequences and adjusted batch sizes")
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    print (f"Data has been converted to numpy arrays")
    trainX = augmentation_operations(trainX, augmentations)
    trainY = augmentation_operations(trainY, augmentations)
    testX = augmentation_operations(testX, augmentations)
    testY = augmentation_operations(testY, augmentations)
    
    print (f"Post augmentation")

    return trainX, trainY, testX, testY
    
@click.command()
@click.option('--folder-read', type=str, default='.', help="Folder where the data is stored.")

def main(folder_read : str) -> None:
            
    # python .\loadandprep.py --folder-read .\datos_npz\
    file_paths = [os.path.join(folder_read, f) for f in os.listdir(folder_read)]

    # augmentations = ['normalize', 'add_noise', 'smooth', 'remove_outliers', 'remove_nans', 'remove_duplicates', 'magnitude_warping', 'scaling', 'time_warping']
    augmentations = ['add_noise']
    
    batch_size = 32
    window_size = 5
    print (f"Dataset will be generated with batch size {batch_size} and window size {window_size}")
    
    model_version = 1
    dataset_version = 1
    # loss mae, mse, huber

    ## CONVOLUTIONAL MODEL---------------------------------------------
    
    print("Convolutional model")
    
    conv_model = Conv1D_model().model
    conv_model.summary()
    metrics_conv=[tf.metrics.MeanAbsoluteError()]
    conv_model.compile(optimizer='adam', loss='mae', metrics=metrics_conv)
    checkpoint_filepath_conv = f'./weights/model_conv_{model_version}_dataset_{dataset_version}_{{loss:.4f}}.weights.h5'
    checkpoint_callback_conv = ModelCheckpoint(
        checkpoint_filepath_conv,
        monitor='loss',            
        mode='min',                    
        save_weights_only=True,        
        save_best_only=True,          
        verbose=1                      
    )
    
    ## LSTM MODEL---------------------------------------------	
    
    print("LSTM model")
    
    lstm_model = Lstm_model().model
    lstm_model.summary()
    metrics_lstm=[tf.metrics.MeanAbsoluteError()]
    lstm_model.compile(optimizer='adam', loss='mae', metrics=metrics_lstm)
    checkpoint_filepath_lstm = f'./weights/model_lstm_{model_version}_dataset_{dataset_version}_{{loss:.4f}}.weights.h5'
    checkpoint_callback_lstm = ModelCheckpoint(
        checkpoint_filepath_lstm,
        monitor='loss',            
        mode='min',                    
        save_weights_only=True,        
        save_best_only=True,          
        verbose=1                      
    )

    ## DENSE MODEL---------------------------------------------
    
    print("Dense model")
    
    dense_model = Dense_model().model
    dense_model.summary()
    metrics_dense=[tf.metrics.MeanAbsoluteError()]
    dense_model.compile(optimizer='adam', loss='mae', metrics=metrics_dense)
    checkpoint_filepath_dense = f'./weights/model_dense_{model_version}_dataset_{dataset_version}_{{loss:.4f}}.weights.h5'
    checkpoint_callback_dense = ModelCheckpoint(
        checkpoint_filepath_dense,
        monitor='loss',            
        mode='min',                    
        save_weights_only=True,        
        save_best_only=True,          
        verbose=1                      
    )
    
    ## Training---------------------------------------------
    
    print("Training")
        
    number_files_per_iteration = 500
    
    for i in range(0, len(file_paths), number_files_per_iteration):
            
        trainX, trainY, testX, testY = dataset_generator(file_paths[i:i+number_files_per_iteration], batch_size, window_size, augmentations)
        
        # print shape of trainX, trainY, testX, testY
        print (f"TrainX shape: {trainX.shape}")
        print (f"TrainY shape: {trainY.shape}")
        print (f"TestX shape: {testX.shape}")
        print (f"TestY shape: {testY.shape}")
        
        conv_model.fit(trainX, trainY, epochs=10, batch_size=32, callbacks=[checkpoint_callback_conv])
        lstm_model.fit(trainX, trainY, epochs=10, batch_size=32, callbacks=[checkpoint_callback_lstm])
        dense_model.fit(trainX, trainY, epochs=10, batch_size=32, callbacks=[checkpoint_callback_dense])
        
    ## EVALUATION---------------------------------------------
    
    print("Evaluation")

    # loss, mae, mse, huber
    
    loss_conv = conv_model.evaluate(testX, testY)
    loss_lstm = lstm_model.evaluate(testX, testY)
    loss_dense = dense_model.evaluate(testX, testY)
    
    print(f"Convolutional model loss: {loss_conv}")
    print(f"LSTM model loss: {loss_lstm}")
    print(f"Dense model loss: {loss_dense}")
    
    # prediction = lstm_model.predict(testX)
    # print(f"prediction 0 :", prediction[0])
    # print(f"testX 0 :", testX[0])
    # print(f"testY 0 :", testY[0])
    
    # # Crear un rango de tiempo para las secuencias
    # time_steps = np.arange(len(testX[0]))
    # future_steps = np.arange(len(testX[0]), len(testX[0]) + len(testY[0]))

    # # Crear la figura y los ejes
    # plt.figure(figsize=(12, 6))

    # # Graficar la entrada (testX)
    # plt.plot(time_steps, testX[0], 'g')

    # # Graficar los valores reales (testY)
    # plt.plot(future_steps, testY[0], 'r')

    # # Graficar las predicciones
    # plt.plot(future_steps, prediction[0], 'b')
    
    # # Añadir títulos y etiquetas
    # plt.title('Predicciones del modelo')
    # plt.xlabel('Paso de Tiempo')
    # plt.ylabel('Valor')

    # # Añadir una leyenda
    # plt.legend()

    # # Mostrar la gráfica
    # plt.show()
    
    return None


if __name__ == '__main__':
    main()
