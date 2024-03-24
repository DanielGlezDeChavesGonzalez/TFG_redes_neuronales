import numpy as np
import tensorflow as tf

# Data Loading Functions
def load_npz(file_path):
    with np.load(file_path) as data:
        return data['timestamps'], data['values']

def _parse_tfrecord_fn(example):
    feature_description = {
        'timestamp': tf.io.FixedLenFeature([], tf.float32),
        'value': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed_features = tf.io.parse_single_example(example, feature_description)
    return parsed_features['timestamp'], parsed_features['value']

def load_tfrecord(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    return raw_dataset.map(_parse_tfrecord_fn)

# Data Augmentation Functions
def augment_data(timestamps, values, augmentations):
    if 'normalize' in augmentations:
        values = (values - tf.reduce_min(values)) / (tf.reduce_max(values) - tf.reduce_min(values))
    if 'add_noise' in augmentations:
        noise = tf.random.normal(shape=tf.shape(values), mean=0.0, stddev=0.05)
        values = values + noise
    return timestamps, values

# Data Generator
def data_generator(file_paths, batch_size, data_format='npz', augmentations=[]):
    if data_format not in ['npz', 'tfrecord']:
        raise ValueError("data_format must be either 'npz' or 'tfrecord'")
    
    if data_format == 'npz':
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(load_npz(x)))
    else:  # 'tfrecord'
        dataset = tf.data.TFRecordDataset(file_paths)
        dataset = dataset.map(_parse_tfrecord_fn)
    
    # Apply data augmentation
    dataset = dataset.map(lambda x, y: augment_data(x, y, augmentations))
    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    return dataset

# Example usage
if __name__ == "__main__":
    # Example file paths
    file_paths = ['data_chunk_0.npz', 'data_chunk_1.npz']  # or TFRecord files
    batch_size = 64
    data_format = 'npz'  # or 'tfrecord'
    augmentations = ['normalize', 'add_noise']  # Data augmentation flags
    
    dataset = data_generator(file_paths, batch_size, data_format, augmentations)

    # Now you can use this dataset directly in your model training
    # model.fit(dataset, epochs=10)
