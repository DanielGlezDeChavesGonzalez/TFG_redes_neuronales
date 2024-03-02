import click
from loguru import logger
import math
from typing import Any, Callable
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns

def read_folder_file(folder_or_file_data: str, headers:bool, separator) -> Any:

    data_array = []
    
    if headers:
        if os.path.isfile(folder_or_file_data):
            data = pd.read_csv(folder_or_file_data, sep=separator, header=0)
            data_array.append(data)
            data.info()

        if os.path.isdir(folder_or_file_data):
            for filename in os.listdir(folder_or_file_data):
                file_path = os.path.join(folder_or_file_data, filename)
            
                if os.path.isfile(file_path):
                    data = pd.read_csv(file_path, sep=separator, header=0)
                else:
                    print(f"'{file_path}' does not exist or is not a file.")
                data_array.append(data)
            data.info()
    
    else:
        if os.path.isfile(folder_or_file_data):
            data = pd.read_csv(folder_or_file_data, sep=separator, header=None)
            data_array.append(data)
            data.info()

        if os.path.isdir(folder_or_file_data):
            for filename in os.listdir(folder_or_file_data):
                file_path = os.path.join(folder_or_file_data, filename)
            
                if os.path.isfile(file_path):
                    data = pd.read_csv(file_path, sep=separator, header=None)
                else:
                    print(f"'{file_path}' does not exist or is not a file.")
                data_array.append(data)
            data.info()
            
    return data_array


def cleaning(data_array : Any , headers:bool) -> Any:
        
    ## Remove duplicates for all files
    data_array = [data.drop_duplicates() for data in data_array]
        
    ## Remove empty strings
    data_array = [data.replace(r'^\s*$', np.nan, regex=True) for data in data_array]

    return data_array

def transformations(data_array: Any , headers:bool) -> Any:
    
    for data in data_array:
        # Convert the second column to float
        data[1] = pd.to_numeric(data[1], errors='coerce')
        
        # Replace Nan values with the mean of the 3 values before and after the Nan value
        data[1] = data[1].fillna(data[1].rolling(3, min_periods=1).mean())
         
    return data_array

def normalization(data_array : Any , headers:bool) :
    
    # Normalize the data of the second column using numpy
    
    for data in data_array:
        
        data[1] = (data[1] - np.min(data[1])) / (np.max(data[1]) - np.min(data[1]))
        
    return data_array

def aumented_data(data_array: Any, parameters: str, headers:bool) -> Any:
        
    match parameters:
        case "jitter":
            for data in data_array:
                data[1] = data[1] + np.random.normal(0, 0.1, len(data[1]))
        case "permutation":
            for data in data_array:
                data[1] = np.random.permutation(data[1])
        case "magnitude_warp":
            for data in data_array:
                data[1] = data[1] * np.random.uniform(0.5, 1.5)
        case _:
            print("Invalid parameter")
    
    return data_array
    
def write_file(data_array: Any, folder_to_write: str, headers: bool, separator:str) -> None:
    
    if not os.path.exists(folder_to_write):
        os.makedirs(folder_to_write)
    
    if headers:
        # Write the data to the folder with the name of the column from the dataframe
        for i in range(len(data_array)):
            print ("escribiendo archivo: " + folder_to_write + "/file" + str(i) + ".csv")
            data_array[i].to_csv(folder_to_write + "/file" + str(i) + ".csv", index=False, sep=separator)
    else:
        # Write the data to the folder without the name of the column from the dataframe
        for i in range(len(data_array)):
            print ("escribiendo archivo: " + folder_to_write + "/file" + str(i) + ".csv")
            data_array[i].to_csv(folder_to_write + "/file" + str(i) + ".csv", header=False, index=False, sep=separator)
            
    return None

def setup_logger(log_to_file: bool, log_level: str) -> None:
    # Configure the logger based on user preferences.

    # Args:
    # log_to_file (bool): If True, logs are written to a file. If False, logs are printed to the console.
    # log_level (str): Level of logging. Example: 'INFO', 'ERROR'.

    # Returns:
    # None
    
    if log_to_file:
        logger.add("debug.log", format="{time} {level} {message}", level=log_level.upper())
    else:
        logger.remove()
        logger.add(lambda msg: click.echo(msg, err=True), format="{time} {level} {message}", level=log_level.upper())

def build_db_uri(db_name: str, host: str = None, port: str = None, user: str = None, password: str = None) -> str:
    # Build a database URI based on the database type and credentials.

    # Args:
    # db_name (str): Name or path of the database.
    # host (str, optional): Host of the database (for PostgreSQL).
    # port (str, optional): Port of the database (for PostgreSQL).
    # user (str, optional): Username for the database (for PostgreSQL).
    # password (str, optional): Password for the database (for PostgreSQL).

    # Returns:
    # str: Database URI.

    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


def load_csv_to_db(csv_file: str, db_uri: str, table_name: str) -> None:
    # Load a CSV file into a database using SQLAlchemy.

    # Args:
    # csv_file (str): Path to the CSV file.
    # db_uri (str): Database URI.
    # table_name (str): Name of the table to which the CSV data will be loaded.

    # Returns:
    # None
    
    try:
        df = pd.read_csv(csv_file)

        engine = create_engine(db_uri)
        with engine.connect() as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"CSV data from '{csv_file}' successfully loaded into table '{table_name}' in database.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
@click.command()
@click.argument('folder_or_file_data', type=str)
@click.argument('folder_to_write', type=str)
@click.option('--headers', type=bool, default=True, help="Input include headers.")
@click.option('--separator', type=str, default=';', help="Separator of the data.")
@click.option('--jitter', is_flag=True, help="Add jitter to the data.")
@click.option('--normalize', is_flag=True, help="Normalize the data.")
@click.option('--permutation', is_flag=True, help="Permute the data.")
@click.option('--magnitude-warp', is_flag=True, help="Magnitude warp the data.")
@click.option('--log-to-file', is_flag=False, help="Enable logging to a file instead of the console.")
@click.option('--log-level', default='INFO', type=str, help="Set the logging level (e.g., INFO, ERROR).")

def main(folder_or_file_data: str, folder_to_write: str, headers: bool, separator :str, jitter: bool, normalize : bool, permutation :bool, magnitude_warp: bool , log_to_file: bool, log_level: str) -> None:

    data = read_folder_file(folder_or_file_data, headers, separator)
    
    # remove the last column of the files
    for i in range(len(data)):
        data[i] = data[i].iloc[:, :-1]
            
    data_modified = data
    
    print ("archivos: " + str(len(data)))
        
    clean_data = cleaning(data_modified, headers)
    write_file(clean_data, folder_to_write + "/cleaned", headers, separator)
    data_modified = clean_data
        
    transformations_data = transformations(data_modified, headers)
    write_file(transformations_data, folder_to_write + "/transformed", headers, separator)
    data_modified = transformations_data
    
    if normalize:
        normalize_data = normalization(normalize_data, headers)
        write_file(normalize_data, folder_to_write + "/normalized", headers, separator)
        data_modified = normalize_data
        
    if jitter:
        jitter_data = aumented_data(data_modified, "jitter", headers)
        write_file(jitter_data, folder_to_write + "/jitter", headers, separator)
        data_modified = jitter_data
    
    if permutation:
        permutation_data = aumented_data(normalize_data, "permutation", headers)
        write_file(permutation_data, folder_to_write + "/permutation", headers, separator)
        data_modified = permutation_data
        
    if magnitude_warp:
        magnitude_warp_data = aumented_data(normalize_data, "magnitude_warp", headers)
        write_file(magnitude_warp_data, folder_to_write + "/magnitude_warp", headers, separator)
        data_modified = magnitude_warp_data
        
    write_file(data_modified, folder_to_write + "/final", headers, separator)


    ## Write the data to a database in postgresql through the docker container
    
    setup_logger(log_to_file, log_level)

    db_uri = build_db_uri(folder_to_write, "localhost", "5432", "postgres", "postgres")

    try:
        
        ## Obtain the name of all the files in the folder to load them up to the database
        for filename in os.listdir(folder_to_write + "/final"):
            csv_file = os.path.join(folder_to_write + "/final", filename)
            table_name = filename.split(".")[0]
            print(f"Loading data from '{csv_file}' into table '{table_name}' in database.")
            load_csv_to_db(csv_file, db_uri, table_name)
            
    except Exception as e:
        click.echo(e)
    ## represent the data in a graph
    # i = 0
    # for data in data_modified:
        
    #     print ("graficando archivo: " + folder_to_write + "/final/file" + str(i) + ".csv")
    #     plt.plot(data[0], data[1])
    #     i+=1
    # plt.show()
    
    # Seaborn graph representation
    for data in data_modified:
        sns.lineplot(data=data)
    plt.show()
    
    return None

if __name__ == '__main__':
    main()
