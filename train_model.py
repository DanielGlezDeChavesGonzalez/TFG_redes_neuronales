
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
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout,Conv1D, MaxPooling1D, Flatten,LSTM # type: ignore
    
def read_data_from_npz(filename):
    # print(f"redadass--------------------- {filename}")
    with np.load(filename) as data:
        # print(f"Data has been successfully loaded from {filename}")
        return data['Timestamp'], data['Value']
     
def augmentation_operations(data, augmentations):
    timestamps, values = data[:, 0], data[:, 1].astype(float)
    if 'normalize' in augmentations:
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
    if 'add_noise' in augmentations:
        noise = np.random.normal(loc=0.0, scale=0.05, size=values.shape)
        values += noise
    if 'smooth' in augmentations:
        values = np.convolve(values, np.ones(5) / 5, mode='same')
    if 'remove_outliers' in augmentations:
        z_scores = np.abs(stats.zscore(values))
        values = values[z_scores < 3]
        timestamps = timestamps[z_scores < 3]
    if 'remove_nans' in augmentations:
        valid_mask = ~np.isnan(values)
        values = values[valid_mask]
        timestamps = timestamps[valid_mask]
    if 'remove_duplicates' in augmentations:
        _, unique_indices = np.unique(values, return_index=True)
        values = values[unique_indices]
        timestamps = timestamps[unique_indices]
    if 'magnitude_warping' in augmentations:
        warping_factors = np.random.uniform(0.9, 1.1, size=values.shape)
        values *= warping_factors
    if 'scaling' in augmentations:
        scaling_factors = np.random.uniform(0.5, 2.0, size=values.shape)
        values *= scaling_factors
    if 'time_warping' in augmentations:
        indices = np.arange(len(values))
        np.random.shuffle(indices)
        values = values[indices]
        timestamps = timestamps[indices]

    augmented_data = np.column_stack((timestamps, values))
    return augmented_data

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
    train_data, test_data = split_data(combined_data, train_ratio)

    trainX, trainY = create_sequences(train_data, batch_size, window_size)
    testX, testY = create_sequences(test_data, batch_size, window_size)

    trainX, trainY = adjust_batch_sizes(trainX, trainY, batch_size, window_size)
    testX, testY = adjust_batch_sizes(testX, testY, batch_size, window_size)

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
# @click.argument('operation', type=str , default='stacionary_and_correlation')
@click.option('--folder-read', type=str, default='.', help="Folder where the data is stored.")
# @click.option('--folder-save', type=str, default='.', help="Folder where the data will be saved.")

def main(folder_read : str) -> None:
            
    # python .\loadandprep.py --folder-read datos_npz
    file_paths = [os.path.join(folder_read, f) for f in os.listdir(folder_read)]

    # augmentations = ['normalize', 'add_noise', 'smooth', 'remove_outliers', 'remove_nans', 'remove_duplicates', 'magnitude_warping', 'scaling', 'time_warping']
    augmentations = ['add_noise']
    
    batch_size = 32
    window_size = 5
    trainX, trainY, testX, testY = dataset_generator(file_paths, batch_size, window_size, augmentations)
        
    # print("------------", trainX[0])
    # print("................" , trainY[0])
    
    # print (f"TrainX shape: {trainX.shape}")
    # print (f"TrainY shape: {trainY.shape}")
    # print (f"TestX shape: {testX.shape}")
    # print (f"TestY shape: {testY.shape}")      
    
    ## CONVOLUTIONAL MODEL---------------------------------------------
    ## ----------------------------------------------------------------
        
    #  model = Sequential()
    # model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    # model.add(Conv1D(64, 3, activation='relu'))
    # model.add(MaxPooling1D(2))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(128, 3, activation='relu'))
    # model.add(Conv1D(128, 3, activation='relu'))
    # model.add(MaxPooling1D(2))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(output_steps))
    
    # conv_model = Sequential([
    #     Conv1D(64, 3, activation='relu', input_shape=(batch_size, window_size)),
    #     Conv1D(64, 3, activation='relu'),
    #     MaxPooling1D(2),
    #     Dropout(0.2),
    #     Conv1D(128, 3, activation='relu'),
    #     Conv1D(128, 3, activation='relu'),
    #     MaxPooling1D(2),
    #     Dropout(0.2),
    #     Flatten(),
    #     Dense(256, activation='relu'),
    #     Dropout(0.2),
    #     Dense(window_size)
    # ])
    
    # metrics_conv=[tf.metrics.MeanAbsoluteError()]
    
    # conv_model.compile(optimizer='adam', loss='mse', metrics=metrics_conv)
    
    # checkpoint_filepath_conv = f'./weights/model_conv_{model_version}_dataset_{dataset_version}_{{loss:.3f}}.weights.h5'
    
    # checkpoint_callback_conv = ModelCheckpoint(
    #     checkpoint_filepath_conv,
    #     monitor='loss',            
    #     mode='min',                    
    #     save_weights_only=True,        
    #     save_best_only=True,          
    #     verbose=1                      
    # )
    
    # conv_model.fit(trainX, trainY, epochs=10, batch_size=32, callbacks=[checkpoint_callback_conv])
            
    # conv_loss = conv_model.evaluate(data_test)
    # print(f"Convolutional model loss: {conv_loss}")
    
    model_version = 1
    dataset_version = 1
    
    ## LSTM MODEL---------------------------------------------	
    ## ----------------------------------------------------------------
    
    lstm_model = Sequential([
        LSTM(64, input_shape=(batch_size,1), return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(256, return_sequences=False),
        Dropout(0.2),
        Dense(window_size)
    ])
    
    lstm_model.summary()
    
    metrics_lstm=[tf.metrics.MeanAbsoluteError()]
    
    lstm_model.compile(optimizer='adam', loss='mse', metrics=metrics_lstm)
    
    checkpoint_filepath_lstm = f'./weights/model_lstm_{model_version}_dataset_{dataset_version}_{{loss:.4f}}.weights.h5'
    
    checkpoint_callback_lstm = ModelCheckpoint(
        checkpoint_filepath_lstm,
        monitor='loss',            
        mode='min',                    
        save_weights_only=True,        
        save_best_only=True,          
        verbose=1                      
    )

    lstm_model.fit(trainX, trainY, epochs=3, batch_size=32, callbacks=[checkpoint_callback_lstm])
    
    ## DENSE MODEL---------------------------------------------
    ## ----------------------------------------------------------------
    
    # dense_model = Sequential([
    #     Dense(64, activation='relu', input_shape=[batch_size, window_size]),
    #     Dropout(0.2),
    #     Dense(128, activation='relu'),
    #     Dropout(0.2),
    #     Dense(256, activation='relu'),
    #     Dropout(0.2),
    #     Dense(window_size)
    # ])
    
    # metrics_dense=[tf.metrics.MeanAbsoluteError()]
    
    # dense_model.compile(optimizer='adam', loss='mse', metrics=metrics_dense)
    
    # checkpoint_filepath_dense = f'./weights/model_dense_{model_version}_dataset_{dataset_version}_{{loss:.3f}}.weights.h5'
    
    # checkpoint_callback_dense = ModelCheckpoint(
    #     checkpoint_filepath_dense,
    #     monitor='loss',            
    #     mode='min',                    
    #     save_weights_only=True,        
    #     save_best_only=True,          
    #     verbose=1                      
    # )
    
    # dense_model.fit(trainX, trainY, epochs=10, batch_size=32, callbacks=[checkpoint_callback_dense])
    
    prediction = lstm_model.predict(testX)
    print(f"prediction 0 :", prediction[0])
    print(f"testX 0 :", testX[0])
    print(f"testY 0 :", testY[0])
    
    # Crear un rango de tiempo para las secuencias
    time_steps = np.arange(len(testX[0]))
    future_steps = np.arange(len(testX[0]), len(testX[0]) + len(testY[0]))

    # Crear la figura y los ejes
    plt.figure(figsize=(12, 6))

    # Graficar la entrada (testX)
    plt.plot(time_steps, testX[0], 'g')

    # Graficar los valores reales (testY)
    plt.plot(future_steps, testY[0], 'r')

    # Graficar las predicciones
    plt.plot(future_steps, prediction[0], 'b')
    
    # Añadir títulos y etiquetas
    plt.title('Predicciones del modelo')
    plt.xlabel('Paso de Tiempo')
    plt.ylabel('Valor')

    # Añadir una leyenda
    plt.legend()

    # Mostrar la gráfica
    plt.show()
    
    return None


if __name__ == '__main__':
    main()
