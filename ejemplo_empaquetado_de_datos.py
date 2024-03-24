import psycopg2
import numpy as np
import tensorflow as tf
# from tensorflow.train import FloatList, Feature, Features, Example

# Database connection parameters
dbname = 'your_database_name'
user = 'your_username'
password = 'your_password'
host = 'your_host'

# Connect to your postgres DB
conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)

# Open a cursor to perform database operations
cur = conn.cursor()

# Query your database
cur.execute("SELECT timestamp, sensor_value FROM registros")
rows = cur.fetchall()

# Close the cursor and connection
cur.close()
conn.close()

# Function to slice data into chunks
def slice_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

# Function to create TFRecord files
# def create_tfrecord(data, filename):
#     with tf.io.TFRecordWriter(filename) as writer:
#         for timestamp, value in data:
#             feature = {
#                 'timestamp': Feature(float_list=FloatList(value=[timestamp.timestamp()])),
#                 'value': Feature(float_list=FloatList(value=[value]))
#             }
#             features = Features(feature=feature)
#             example = Example(features=features)
#             writer.write(example.SerializeToString())

# Function to create NPZ files
def create_npz(data, filename):
    timestamps = [x[0].timestamp() for x in data]
    values = [x[1] for x in data]
    np.savez(filename, timestamps=timestamps, values=values)

# Main process
chunk_size = 10000  # Define your chunk size
sliced_data = slice_data(rows, chunk_size)

for idx, chunk in enumerate(sliced_data):
    tfrecord_filename = f'data_chunk_{idx}.tfrecord'
    npz_filename = f'data_chunk_{idx}.npz'
    
    # create_tfrecord(chunk, tfrecord_filename)
    create_npz(chunk, npz_filename)

print("Data has been successfully saved in TFRecord and NPZ formats.")
