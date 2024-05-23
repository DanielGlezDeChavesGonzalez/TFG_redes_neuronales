
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

def adjust_batch_sizes(X, Y, expected_size_X, expected_size_Y):
    adjusted_X = []
    adjusted_Y = []
    
    for x, y in zip(X, Y):
        if len(x) == expected_size_X and len(y) == expected_size_Y:
            adjusted_X.append(x)
            adjusted_Y.append(y)
    
    return adjusted_X, adjusted_Y

    
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
                
    # print (f"TrainX last: {trainX[len(trainX)-1]}")
    # print (f"TrainY last: {trainY[len(trainY)-1]}")
    # print (f"TestX last: {testX[len(testX)-1]}")
    # print (f"TestY last: {testY[len(testY)-1]}")
    
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
        
    # conv_model = tf.keras.Sequential([
    #     tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    #     tf.keras.layers.Dense(units=32, activation='relu'),
    #     tf.keras.layers.Dense(units=1)
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

    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(batch_size, 1)),
        tf.keras.layers.LSTM(units=50),
        tf.keras.layers.Dense(units=8),
        tf.keras.layers.Dense(units=window_size)
    ])
    
    lstm_model.summary()
    
    metrics_lstm=[tf.metrics.MeanAbsoluteError()]
    
    lstm_model.compile(optimizer='adam', loss='mse', metrics=metrics_lstm)
    
    checkpoint_filepath_lstm = f'./weights/model_lstm_{model_version}_dataset_{dataset_version}_{{loss:.3f}}.weights.h5'
    
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
        
    dense_model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, activation='relu', input_shape=(batch_size,)),
        tf.keras.layers.Dense(units=window_size)
    ])
    
    metrics_dense=[tf.metrics.MeanAbsoluteError()]
    
    dense_model.compile(optimizer='adam', loss='mse', metrics=metrics_dense)
    
    checkpoint_filepath_dense = f'./weights/model_dense_{model_version}_dataset_{dataset_version}_{{loss:.3f}}.weights.h5'
    
    checkpoint_callback_dense = ModelCheckpoint(
        checkpoint_filepath_dense,
        monitor='loss',            
        mode='min',                    
        save_weights_only=True,        
        save_best_only=True,          
        verbose=1                      
    )
    
    dense_model.fit(trainX, trainY, epochs=10, batch_size=32, callbacks=[checkpoint_callback_dense])
    
    prediction = lstm_model.predict(testX)
    print(f"prediction 0 :", prediction[0])
    print(f"testX 0 :", testX[0])
    print(f"testY 0 :", testY[0])
    
    # print(f"Prediction shape: {prediction.shape}")
    # que la entrada se vea en otro color (verde)
    # plt.plot(testX, color='green')
    # plt.plot( testY, color='red')
    # # que la predicción se ve al final de la serie temporal en otro color (azul)
    # plt.plot(prediction, color='blue')
    # plt.legend(['Input', 'Real', 'Prediction'])
    # plt.title('Model prediction')
    # plt.ylabel('Value')
    # plt.xlabel('Index')
    # plt.show()
    
    return None


if __name__ == '__main__':
    main()
