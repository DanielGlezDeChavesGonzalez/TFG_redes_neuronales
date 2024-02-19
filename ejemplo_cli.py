import click
from loguru import logger
import math
from typing import Any, Callable
import os
import numpy as np
import pandas as pd

def read_folder(folder_to_read: str) -> Any:

    data_array = []

    for filename in os.listdir(folder_to_read):
        file_path = os.path.join(folder_to_read, filename)
    
        if os.path.isfile(file_path):
            data = pd.read_csv(file_path, sep=';', header=None)
        else:
            print(f"'{file_path}' does not exist or is not a file.")
        data_array.append(data)
        
    data.info()
    return data_array

def cleaning(data_array : Any ) -> Any:
        
    ## Remove duplicates for all files
    data_array = [data.drop_duplicates() for data in data_array]
        
    ## Remove empty strings
    data_array = [data.replace(r'^\s*$', np.nan, regex=True) for data in data_array]

    return data_array

def transformations(data_array: Any) -> Any:
    
    # Fill in the missing values with the mean of the 3 previous values and 3 following values
    data_array = [data.fillna(data.rolling(window=7, min_periods=1).mean()) for data in data_array]
    
    return data_array

def normalize(data_array : Any) -> Any:
    
    # Normalize the data
    data_array = [data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) for data in data_array]
    
    return data_array

def unnormalize(data_array : Any) -> Any:
        
    # Unnormalize the data
    data_array = [data.apply(lambda x: (x * (np.max(x) - np.min(x))) + np.min(x)) for data in data_array]
        
    return data_array

def aumented_data(data_array: Any) -> Any:
        
    # Aumented data
    data_array = [data.append(data.rolling(window=7, min_periods=1).mean()) for data in data_array]
        
    return data_array
    
def write_file(data_array: Any, folder_to_write: str) -> None:
    
    if not os.path.exists(folder_to_write):
        os.makedirs(folder_to_write)
        
    for i in range(len(data_array)):
        data_array[i].to_csv(folder_to_write + "/file_" + str(i) + ".csv", sep=';', index=False)
        
    return None


@click.command()
@click.argument('folder_to_read', type=str)
@click.argument('folder_to_write', type=str)
# @click.option('--log-to-file', is_flag=True, help="Enable logging to a file instead of the console.")
def main(folder_to_read: str, folder_to_write: str) -> None:

    data = read_folder(folder_to_read)
    
    print ("archivos: " + str(len(data)))
        
    clean_data = cleaning(data)
        
    transformations_data = transformations(clean_data)
    
    write_file(transformations_data, folder_to_write + "/transformed")
    
    normalize_data = normalize(transformations_data)
    
    write_file(normalize_data, folder_to_write + "/normalized")

    unnormalize_data = unnormalize(normalize_data)
    
    write_file(unnormalize_data, folder_to_write + "/unnormalized")
    
    return None

if __name__ == '__main__':
    main()
