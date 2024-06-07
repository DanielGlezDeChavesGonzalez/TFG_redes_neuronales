import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from scipy import stats
import os
import click
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from model_creators import Lstm_model, Conv1D_model, Dense_model
import tensorflow as tf

def augmentation_operations(data, augmentations):
    
    for augmentation in augmentations:
        if augmentation == 'normalize':
            data = (data - data.mean()) / data.std()
        elif augmentation == 'add_noise':
            data = data + np.random.normal(0, 0.1, data.shape)
        elif augmentation == 'smooth':
            data = data.rolling(window=5).mean()
        elif augmentation == 'remove_outliers':
            data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
        elif augmentation == 'remove_nans':
            data = data.dropna()
        elif augmentation == 'remove_duplicates':
            data = data.drop_duplicates()
        elif augmentation == 'magnitude_warping':
            data = data * np.random.uniform(0.9, 1.1)
        elif augmentation == 'scaling':
            data = data * np.random.uniform(0.5, 1.5)
        elif augmentation == 'time_warping':
            data = data.sample(frac=1).reset_index(drop=True)
            
    return data


# Función para dividir los datos en entrenamiento y prueba
def split_data(file_list, train_ratio=0.9):
    print("Split data")
    all_data = []
    for file in file_list:
        data = np.load(file)
        # timestamps = data['Timestamp']
        values = data['Value']
        aug_data = augmentation_operations(values, ['add_noise'])
        all_data.extend(aug_data)
    
    all_data = np.array(all_data)
    
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    return train_data, test_data

# Generador de datos
def data_generator(data, batch_size, seq_length, pred_length):
    print("Data generator")
    X_batch, y_batch = [], []
    
    for i in range(len(data) - seq_length - pred_length):
        
        X_batch.append(data[i:i+seq_length])
        y_batch.append(data[i+seq_length:i+seq_length+pred_length])
        
        if len(X_batch) == batch_size:
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            # print(f"X_batch shape: {X_batch.shape}")
            # print(f"y_batch shape: {y_batch.shape}")
            yield X_batch, y_batch
            X_batch, y_batch = [], []
    
    if X_batch:
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        # print(f"X_batch shape: {X_batch.shape}")
        # print(f"y_batch shape: {y_batch.shape}")
        yield X_batch, y_batch

@click.command()
@click.option('--folder-read', type=str, default='.', help="Folder where the data is stored.")

def main(folder_read : str) -> None:
            
    # python .\train_model.py --folder-read ..\datos_npz\
    file_paths = [os.path.join(folder_read, f) for f in os.listdir(folder_read)]
    
    train_data, test_data = split_data(file_paths)

    # Definir el modelo LSTM
    # Entrenar el modelo
    batch_size = 32
    seq_length = 32
    pred_length = 5
    model_version = 1
    
    print("LSTM model")
    
    # lstm_model = Lstm_model(pred_length).model
    # lstm_model.summary()
    # metrics_lstm=[tf.metrics.MeanAbsoluteError()]
    # lstm_model.compile(optimizer='adam', loss='mse', metrics=metrics_lstm)
    
    model = Sequential()
    model.add(LSTM(64, input_shape=(32, 1)))
    model.add(Dense(5))
    
    model.summary()
    
    model.compile(optimizer='adam', loss='mse', metrics=[tf.metrics.MeanAbsoluteError()])
    checkpoint_filepath_lstm = f'../weights/model_lstm_modelversion_{model_version}_outputs_{pred_length}_{{loss:.4f}}.weights.h5'
    checkpoint_callback_lstm = ModelCheckpoint(
        checkpoint_filepath_lstm,
        monitor='loss',            
        mode='min',                    
        save_weights_only=True,        
        save_best_only=True,          
        verbose=1                      
    )


    train_generator = data_generator(train_data, batch_size, seq_length, pred_length)
    
    steps_per_epoch = len(train_data) // batch_size
    model.fit(train_generator, epochs=10, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback_lstm])

    # model.fit(train_generator, epochs=10 , callbacks=[checkpoint_callback_lstm])

    # Evaluar el modelo en los datos de prueba
    test_generator = data_generator(test_data, batch_size, seq_length, pred_length)
    test_loss = model.evaluate(test_generator)
    print(f"Test loss: {test_loss}")
    
if __name__ == '__main__':
    main()
