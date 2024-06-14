
import numpy as np
import tensorflow as tf
import click
from scipy import stats
import os
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from matplotlib import pyplot as plt


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
    
    with np.load(file_path, allow_pickle=True) as data:
        timestamps = data['Timestamp']
        values = data['Value']
        all_timestamps.extend(timestamps)
        all_values.extend(values)
        
    return np.array(list(zip(all_timestamps, all_values)))

def create_sequences(data, window_size, n_outputs):
    X, Y = [], []
    for i in range(len(data) - window_size - n_outputs + 1):
        X.append(data[i:i + window_size, 1].reshape(window_size, 1))  # Reshape to (window_size, 1)
        Y.append(data[i + window_size:i + window_size + n_outputs, 1])  # Sequence of length n_outputs as target
    X, Y = np.array(X), np.array(Y)
    #print(f"create_sequences -> X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y

def data_generator(file_paths, batch_size, window_size, n_outputs, augmentations=[]):
    while True:
        for file in file_paths:
            data = read_and_combine_data(file)
            X, Y = create_sequences(data, window_size, n_outputs)
            #print(f"data_generator -> X shape: {X.shape}, Y shape: {Y.shape}")

            for i in range(0, len(X) - batch_size + 1, batch_size):
                batch_X = X[i:i + batch_size]
                batch_Y = Y[i:i + batch_size]
                batch_X = augmentation_operations(batch_X, augmentations)
                batch_Y = augmentation_operations(batch_Y.reshape(-1, n_outputs), augmentations)  # No need to flatten

                # print(f"data_generator -> batch_X shape: {batch_X.shape}, batch_Y shape: {batch_Y.shape}")
                # print(f"batch_X: {batch_X}")
                # print(f"batch_Y: {batch_Y}")
                yield batch_X, batch_Y
                    
@click.command()
@click.option('--folder-read', type=str, default='./datos_npz/', help="Folder where the data is stored.")

def main(folder_read: str) -> None:
    banc = "banc_1.csv"
    
    file_paths = [os.path.join(folder_read, f) for f in os.listdir(folder_read)]
    file_paths = [f for f in file_paths if banc in f]
    print(f"File paths: {file_paths}")    
    
    augmentations = ['add_noise']
    batch_size = 32
    window_size = 32
    n_outputs = 5
    print(f"Dataset will be generated with batch size {batch_size} and window size {n_outputs}")

    model_version = 1
    dataset_version = 1
    
    print("Dense model")
    dense_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(32, 1)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(n_outputs)
        ])
    dense_model.summary()
    
    metrics_dense = [tf.metrics.MeanAbsoluteError()]
    dense_model.compile(optimizer='adam', loss='mse', metrics=metrics_dense)
    checkpoint_filepath_dense = f'../weights/model_dense{banc}_modelversion_{model_version}_outputs_{n_outputs}_{{loss:.10f}}.weights.h5'
    checkpoint_callback_dense = ModelCheckpoint(
        checkpoint_filepath_dense,
        monitor='loss',
        mode='min',
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    )

    num_epochs = 25
    print("Training")
    train_gen = data_generator(file_paths, batch_size, window_size, n_outputs, augmentations)
    
    steps_per_epoch = (sum([len(read_data_from_npz(f)[1]) for f in file_paths]) // batch_size) // num_epochs

    print(f"Steps per epoch: {steps_per_epoch}")

    for batch_X, batch_Y in train_gen:
        print(f"Training batch -> batch_X shape: {batch_X.shape}, batch_Y shape: {batch_Y.shape}")
        break

    history = dense_model.fit(train_gen, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback_dense])

    print("Evaluation")
    test_gen = data_generator(file_paths, batch_size, window_size, n_outputs, augmentations)
    for batch_X, batch_Y in test_gen:
        print(f"Evaluation batch -> batch_X shape: {batch_X.shape}, batch_Y shape: {batch_Y.shape}")
        break

    loss_dense = dense_model.evaluate(test_gen, steps=steps_per_epoch)
    print(f"dense model loss: {loss_dense}")

    print("Prediction")
    test_gen = data_generator(file_paths, batch_size, window_size, n_outputs, augmentations)
    testX, testY = next(test_gen)

    # Reshape the input to have the correct shape for dense model prediction
    dummy = testX[0].reshape((1, 32, 1))  # (1, sequence_length, 1)


    print(f"TestX: {testX[0].flatten()}")
    print(f"TestX shape: {testX[0].shape}")
    print(f"dummy: {dummy.flatten()}")
    print(f"dummy shape: {dummy.shape}")

    prediction_dense = dense_model.predict(dummy)
    prediction_dense = prediction_dense.flatten()  # Flatten the prediction to match true values shape

    print(f"Dnese model prediction: {prediction_dense}")
    print(f"True: {testY[0]}")
    
    print(f"loss per epoch: {history.history['loss']}")

    ## print model loss progression
    plt.plot(history.history['loss'])
    plt.title('Model loss: '+ banc)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    return None

if __name__ == '__main__':
    main()