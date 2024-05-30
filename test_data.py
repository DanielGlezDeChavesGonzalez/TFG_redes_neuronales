from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import click
from arch.unitroot import PhillipsPerron
import os
from loguru import logger
import numpy as np

# augmented dickey fuller test represented with graphs with plt
def adf_test(data: pd.Series) -> None:
    print("ADF Test-----------------------------------------")
    values = data['Value']
    
    # Realizar la prueba de Dickey-Fuller en la columna 'Value'
    result = adfuller(values)
    
    # Imprimir los resultados
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Kwiatkowski-Phillips-Schmidt-Shin test represented with graphs with plt
def kpss_test(data: pd.Series) -> None:
    print("KPSS Test-----------------------------------------")
    values = data['Value']
    
    # Realizar la prueba de KPSS en la columna 'Value'
    result = kpss(values)
    
    # Imprimir los resultados
    print(f'KPSS Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[3].items():
        print(f'\t{key}: {value}')
        

# Autocorrelation and Partial Autocorrelation Function represented with graphs with plt
def acf_pacf(data: pd.Series) -> None:
    print("ACF and PACF-----------------------------------------")
    values = data['Value']
    # Crear una figura con un panel dividido en 1 fila y 2 columnas
    plt.figure(figsize=(12, 5))
    # Gráfico de la función de autocorrelación
    plt.subplot(2, 1, 1)
    plot_acf(values, ax=plt.gca())
    plt.title(f'ACF')
    # Gráfico de la función de autocorrelación parcial
    plt.subplot(2, 1, 2)
    plot_pacf(values, ax=plt.gca())
    plt.title(f'PACF')
    # Ajustar los gráficos
    plt.subplots_adjust(hspace=0.5)
    # Mostrar los gráficos
    plt.show()


# Phillips-Perron test represented with graphs with plt
def pp_test(data: pd.Series) -> None:
    print("PP Test-----------------------------------------")
    values = data['Value']
    
    # Realizar la prueba de Phillips-Perron en la columna 'Value'
    result = PhillipsPerron(values)
    
    # Imprimir los resultados
    print(f'PP Statistic: {result.stat}')
    print(f'p-value: {result.pvalue}')
    print(f'Critical Values: {result.critical_values}')
    print(f'Null Hypothesis: {result.null_hypothesis}')
    print(f'Alternative Hypothesis: {result.alternative_hypothesis}')

            
    return None

# def read_data_from_npz(filename):
#     with np.load(filename) as data:
#         return data['Timestamp'], data['Value']
def read_data_from_npz(filename, data_type=np.float32):
    print(f"Reading data from {filename}")
    with np.load(filename) as data:
        timestamps = data['Timestamp']
        values = data['Value'].astype(data_type)
        return timestamps, values

@click.command()
@click.option('--folder-read', type=str, default='.', help="Folder where the data is stored.")

def main(folder_read : str) -> None:
    
    all_timestamps = []
    all_values = []
    
    data_type = np.float32
        
    file_paths = [os.path.join(folder_read, f) for f in os.listdir(folder_read)]

    for file in file_paths:
        timestamps, values = read_data_from_npz(file , data_type)
        all_timestamps.extend(timestamps)
        all_values.extend(values)
        
        
    df = pd.DataFrame({'Timestamp': all_timestamps, 'Value': all_values})
    print(df.head())
    adf_test(df)
    kpss_test(df)
    acf_pacf(df)
    pp_test(df)
    
if __name__ == '__main__':
    main()
    