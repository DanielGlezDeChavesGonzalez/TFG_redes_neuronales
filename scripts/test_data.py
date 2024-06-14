from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import click
from arch.unitroot import PhillipsPerron
import os
from loguru import logger
import numpy as np

def read_data_from_csv(filename):
    df = pd.read_csv(filename, sep=';', names=['Timestamp', 'Value'], 
                     float_precision='high', dtype={'Timestamp': 'int32', 'Value': 'float32'})
    
    return df['Timestamp'].values, df['Value'].values

def write_results_to_txt(results, file_save):
    with open(file_save, 'a') as file:
        file.write(results + '\n')

def process_adf( filename, value, file_save):
    print(f"Processing file {filename}")

    results_adf = adfuller(value)
    
    results = (
        f"Filename: {filename}\n"
        f"ADF Test:\nADF Statistic: {results_adf[0]}\n"
        f"p-value: {results_adf[1]}\n"
        f"Critical Values: {results_adf[4]}\n"
    )
    print("Results: ", results)

    write_results_to_txt(results, file_save)
    
def process_pp( filename, value, file_save):
    print(f"Processing file {filename}")

    results_pp = PhillipsPerron(value)

    results = (
        f"Filename: {filename}\n"
        f"Phillips-Perron Test:\n"
        f"PP Statistic: {results_pp.stat}\n"
        f"p-value: {results_pp.pvalue}\n"
        f"Critical Values: {results_pp.critical_values}\n"
    )
    print("Results: ", results)

    write_results_to_txt(results, file_save)
    
def process_kpss( filename, value, file_save):
    print(f"Processing file {filename}")

    results_kpss = kpss(value)

    results = (
        f"Filename: {filename}\n"
        f"KPSS Test:\n"
        f"KPSS Statistic: {results_kpss[0]}\n"
        f"p-value: {results_kpss[1]}\n"
        f"Critical Values: {results_kpss[3]}\n"
    )
    print("Results: ", results)

    write_results_to_txt(results, file_save)

@click.command()
@click.option('--folder-read', type=str, default='.', help="Folder where the data is stored.")
@click.option('--file-save', type=str, default='results.txt', help="File where the results are saved.")
def main(folder_read: str, file_save: str) -> None:
    if not folder_read:
        logger.error("No folder was provided")
        return

    if not file_save:
        logger.error("No file was provided")
        return

    file_paths = [os.path.join(folder_read, f) for f in os.listdir(folder_read) if f.endswith('.csv')]
    
    print(f"Reading data from {len(file_paths)} files")
    
    if not os.path.exists(file_save):
        open(file_save, 'w').close()
        
    for file in file_paths:
        print("Processing file adf: ", file)
        timestamp, value = read_data_from_csv(file)
        process_adf(file, value, file_save)
        # eight_parts = np.array_split(value, 16)
        
        # for i in range(4, 8):
        #     process_adf(file + f"_part_{i}", eight_parts[i], file_save)
        
    for file in file_paths:
        print("Processing file pp: ", file)
        timestamp, value = read_data_from_csv(file)
        process_kpss(file, value, file_save)
        # eight_parts = np.array_split(value, 16)
        
        # for i in range(4, 8):
        #     process_pp(file + f"_part_{i}", eight_parts[i], file_save)
        
        
                
    for file in file_paths:
        print("Processing file kpss: ", file)
        timestamp, value = read_data_from_csv(file)
        process_pp(file, value, file_save)
        # eight_parts = np.array_split(value, 16)
        
        # for i in range(4, 8):
        #     process_kpss(file + f"_part_{i}", eight_parts[i], file_save)
        
    print("Results saved")

if __name__ == '__main__':
    main()

