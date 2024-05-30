import psycopg2
from loguru import logger
import numpy as np
import pandas as pd
import os
from typing import Any, Callable
import click
from scipy import stats


# Database connection parameters
dbname = "datos_temporales"
host = "postgres"
port = "5432"
user = "postgres"
password = "postgres"

def cleaning(data_array : Any) -> Any:
        
    ## Remove duplicates for all files
    data_array = [data.drop_duplicates() for data in data_array]
        
    ## Remove empty strings
    data_array = [data.replace(r'^\s*$', np.nan, regex=True) for data in data_array]
    
    ## Remove outliers from the second column until last using the z-score method
    data_array = [data[(np.abs(stats.zscore(data)) < 3).all(axis=1)] for data in data_array]

    return data_array

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
    
    # f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
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

@click.command()
# @click.argument('operation', type=str , default='stacionary_and_correlation')
@click.option('--folder-read', type=str, default='.', help="Folder where the data is stored.")
@click.option('--folder-save', type=str, default='.', help="Folder where the data will be saved.")

def main(folder_read : str, folder_save: str) -> None:
    
    # print(f"Data will be saved in {folder_save}")

    if folder_read:
        # python .\load_data.py --folder-read .\datos_sensores_prueba\
        logger.info(f"Data will be loaded from {folder_read}")
        data = load_data_from_folder(folder_read)
    else:
        # python .\load_data.py
        logger.info(f"Data will be loaded from the database")
        data = load_data_from_database()
        
    data = cleaning(data)
        
    # python .\load_data.py
    chunk_size = 10000
    unified_data = pd.concat(data)
    # sliced_data = slice_data(data, chunk_size)
    sliced_data = slice_data(unified_data, chunk_size)
    for idx, chunk in enumerate(sliced_data):
        npz_filename = f'data_chunk_{idx}.npz'
        create_npz(chunk, npz_filename, folder_save)
    print("Data has been successfully saved in NPZ format.")
    
if __name__ == '__main__':
    main()
