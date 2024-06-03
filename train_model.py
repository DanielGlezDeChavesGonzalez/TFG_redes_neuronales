
import numpy as np
import tensorflow as tf
import click
from scipy import stats
import os
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from model_creators import Lstm_model, Conv1D_model, Dense_model
from matplotlib import pyplot as plt
# from loguru import logger
# from typing import Any, Callable
# import pandas as pd
# from sqlalchemy import create_engine
# import seaborn as sns
# import IPython
# import IPython.display
# import dask.dataframe as dd
# from tensorflow.keras.models import Sequential # type: ignore

def augmentation_operations(data, augmentations):
    augmented_data = np.array(data.copy())
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

def read_and_combine_data(file_path):
    all_timestamps = []
    all_values = []

    timestamps, values = read_data_from_npz(file_path)
    all_timestamps.extend(timestamps)
    all_values.extend(values)
        
    return np.array(list(zip(all_timestamps, all_values)))

def split_data(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    return data[:train_size], data[train_size:]

def create_sequences(data, batch_size, window_size):
    X, Y = [], []
    ## que no haya sobreposicion
    # print (f"Data: {data}")
    # print (f"Data len: {len(data)}")
    # print (f"Window size: {window_size}")
    # print (f"Batch size: {batch_size}")
    for i in range(0, len(data) - window_size - batch_size, batch_size):
        # print (f"i: {i}")
        X.append(data[i:i + batch_size, 1])
        # print (f"X: {X}")
        Y.append(data[i + batch_size:i + batch_size + window_size, 1])
        # print (f"Y: {Y}")
    return X, Y

def adjust_batch_sizes(X, Y, expected_size_X, expected_size_Y):
    adjusted_X = []
    adjusted_Y = []

    for x, y in zip(X, Y):
        if len(x) == expected_size_X and len(y) == expected_size_Y:
            adjusted_X.append(x)
            adjusted_Y.append(y)

    return adjusted_X, adjusted_Y
    
def data_generator(file_paths, batch_size, window_size, augmentations=[]):
    for file in file_paths:
        data = read_and_combine_data(file)
        
        X, Y = create_sequences(data, batch_size, window_size)

        X, Y = adjust_batch_sizes(X, Y, batch_size, window_size)

        X = augmentation_operations(X, augmentations)
        Y = augmentation_operations(Y, augmentations)

        for i in range(0, len(X), batch_size):
            # print(f"Input shape: {X[i:i + batch_size].shape}, Target shape: {Y[i:i + batch_size].shape}")
            yield X[i:i + batch_size], Y[i:i + batch_size]
            
@click.command()
@click.option('--folder-read', type=str, default='.', help="Folder where the data is stored.")

def main(folder_read : str) -> None:
            
    # python .\loadandprep.py --folder-read .\datos_npz\
    file_paths = [os.path.join(folder_read, f) for f in os.listdir(folder_read)]
        
    # augmentations = ['normalize', 'add_noise', 'smooth', 'remove_outliers', 'remove_nans', 'remove_duplicates', 'magnitude_warping', 'scaling', 'time_warping']
    augmentations = ['add_noise']
    
    batch_size = 32
    n_outputs = 5
    # n_outputs = 10
    print (f"Dataset will be generated with batch size {batch_size} and window size {n_outputs}")
    
    model_version = 1
    dataset_version = 1
    # loss mae, mse, huber

    ## CONVOLUTIONAL MODEL---------------------------------------------
    
    print("Convolutional model")
    
    conv_model = Conv1D_model(n_outputs).model
    conv_model.summary()
    metrics_conv=[tf.metrics.MeanAbsoluteError()]
    conv_model.compile(optimizer='adam', loss='mse', metrics=metrics_conv)
    checkpoint_filepath_conv = f'./weights/model_conv_modelversion_{model_version}_outputs_{n_outputs}_{{loss:.4f}}.weights.h5'
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
    
    lstm_model = Lstm_model(n_outputs).model
    lstm_model.summary()
    metrics_lstm=[tf.metrics.MeanAbsoluteError()]
    lstm_model.compile(optimizer='adam', loss='mse', metrics=metrics_lstm)
    checkpoint_filepath_lstm = f'./weights/model_lstm_modelversion_{model_version}_outputs_{n_outputs}_{{loss:.4f}}.weights.h5'
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
    
    dense_model = Dense_model(n_outputs).model
    dense_model.summary()
    metrics_dense=[tf.metrics.MeanAbsoluteError()]
    dense_model.compile(optimizer='adam', loss='mse', metrics=metrics_dense)
    checkpoint_filepath_dense = f'./weights/model_dense_modelversion_{model_version}_outputs_{n_outputs}_{{loss:.4f}}.weights.h5'
    checkpoint_callback_dense = ModelCheckpoint(
        checkpoint_filepath_dense,
        monitor='loss',            
        mode='min',                    
        save_weights_only=True,        
        save_best_only=True,          
        verbose=1                      
    )
    
    ## Training---------------------------------------------
    
    num_epochs = 25
        
    print("Training")
    train_gen = data_generator(file_paths, batch_size, n_outputs, augmentations)
    steps_per_epoch = sum([len(read_data_from_npz(f)) for f in file_paths]) // batch_size
    
    print(f"Steps per epoch: {steps_per_epoch}")
    
    conv_model.fit(train_gen, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback_conv])
    lstm_model.fit(train_gen, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback_lstm])
    dense_model.fit(train_gen, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback_dense])

    ## EVALUATION---------------------------------------------
    print("Evaluation")
    test_gen = data_generator(file_paths, batch_size, n_outputs, augmentations)

    loss_conv = conv_model.evaluate(test_gen, steps=steps_per_epoch)
    print(f"Convolutional model loss: {loss_conv}")
    
    loss_lstm = lstm_model.evaluate(test_gen, steps=steps_per_epoch)
    print(f"LSTM model loss: {loss_lstm}")
    
    loss_dense = dense_model.evaluate(test_gen, steps=steps_per_epoch)
    print(f"Dense model loss: {loss_dense}")
    
    ## PREDICTION---------------------------------------------
    print("Prediction")
    test_gen = data_generator(file_paths, batch_size, n_outputs, augmentations)
    testX, testY = next(test_gen)
    
    prediction_conv = conv_model.predict(testX)
    prediction_lstm = lstm_model.predict(testX)
    prediction_dense = dense_model.predict(testX)
    
    print(f"Convolutional model prediction: {prediction_conv}")
    print(f"LSTM model prediction: {prediction_lstm}")
    print(f"Dense model prediction: {prediction_dense}")
    
    ## PLOTTING---------------------------------------------
    
    # Asumiendo que testX, testY, prediction_conv, prediction_lstm, y prediction_dense son arrays de numpy
    length = len(testX[0])
    x = np.arange(length, 2*length)

    plt.figure(figsize=(20, 10))
    plt.plot(testX[0], label='Input')

    # Concatenar testX con testY y las predicciones
    plt.plot(np.concatenate([testX[0], testY[0]]), label='True')
    plt.plot(np.concatenate([testX[0], prediction_conv[0]]), label='Prediction_conv')
    plt.plot(np.concatenate([testX[0], prediction_lstm[0]]), label='Prediction_lstm')
    plt.plot(np.concatenate([testX[0], prediction_dense[0]]), label='Prediction_dense')

    plt.legend()
    plt.show()
    return None


if __name__ == '__main__':
    main()
