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
def adf_test(data: pd.Series):
    # print("ADF Test-----------------------------------------")
    values = data['Value']
    
    # Realizar la prueba de Dickey-Fuller en la columna 'Value'
    result = adfuller(values)
    
    # # Imprimir los resultados
    # print(f'ADF Statistic: {result[0]}')
    # print(f'p-value: {result[1]}')
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print(f'\t{key}: {value}')
        
    return result

# Kwiatkowski-Phillips-Schmidt-Shin test represented with graphs with plt
def kpss_test(data: pd.Series):
    # print("KPSS Test-----------------------------------------")
    values = data['Value']
    
    # Realizar la prueba de KPSS en la columna 'Value'
    result = kpss(values)
    
    # Imprimir los resultados
    # print(f'KPSS Statistic: {result[0]}')
    # print(f'p-value: {result[1]}')
    # print('Critical Values:')
    # for key, value in result[3].items():
    #     print(f'\t{key}: {value}')
        
    return result
        
# Phillips-Perron test represented with graphs with plt
def pp_test(data: pd.Series):
    # print("PP Test-----------------------------------------")
    values = data['Value']
    
    # Realizar la prueba de Phillips-Perron en la columna 'Value'
    result = PhillipsPerron(values)
    
    # Imprimir los resultados
    # print(f'PP Statistic: {result.stat}')
    # print(f'p-value: {result.pvalue}')
    # print(f'Critical Values: {result.critical_values}')
    # print(f'Null Hypothesis: {result.null_hypothesis}')
    # print(f'Alternative Hypothesis: {result.alternative_hypothesis}')
            
    return result

# Autocorrelation and Partial Autocorrelation Function represented with graphs with plt
# def acf_pacf(data: pd.Series):
    # print("ACF and PACF-----------------------------------------")
    # values = data['Value']
    # Crear una figura con un panel dividido en 1 fila y 2 columnas
    # plt.figure(figsize=(12, 5))
    # # Gráfico de la función de autocorrelación
    # plt.subplot(2, 1, 1)
    # plot_acf(values, ax=plt.gca())
    # plt.title(f'ACF')
    # # Gráfico de la función de autocorrelación parcial
    # plt.subplot(2, 1, 2)
    # plot_pacf(values, ax=plt.gca())
    # plt.title(f'PACF')
    # # Ajustar los gráficos
    # plt.subplots_adjust(hspace=0.5)
    # # Mostrar los gráficos
    # plt.show()
    
    # Devolver los valores de la función de autocorrelación y autocorrelación parcial
    # return plot_acf(values), plot_pacf(values)

def read_data_from_csv(filename):
    print(f"Reading data from {filename}")
    data = pd.read_csv(filename, sep=';', names=['Timestamp', 'Value'], float_precision='high')
    return data['Timestamp'], data['Value']

def write_results_to_txt(results, filename):
    
    # Si el archivo no existe, se crea
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(results)
    # Si el archivo ya existe, se sobreescribe
    
    else:
        with open(filename, 'w') as f:
            f.write(results)
            
    print(f"Results saved in {filename}")
        

@click.command()
@click.option('--folder-read', type=str, default='.', help="Folder where the data is stored.")
@click.option('--file-save', type=str, default='.', help="File where the results are saved.")

def main(folder_read : str , file_save : str) -> None:
    
    if not folder_read:
        logger.error("No folder was provided")
        return
    
    if not file_save:
        logger.error("No file was provided")
        return
    
    # Ejecutar el script con los siguientes argumentos
    # python test_data.py --folder-read .\datos_sensores\ --file-save resultados_test.txt
    
    all_timestamps = []
    all_values = []
            
    file_paths = [os.path.join(folder_read, f) for f in os.listdir(folder_read)]

    for file in file_paths:
        timestamps, values = read_data_from_csv(file)
        all_timestamps.extend(timestamps)
        all_values.extend(values)
        
    print(f"Data read from {len(file_paths)} files")
    
    df = pd.DataFrame({'Timestamp': all_timestamps, 'Value': all_values})
    # print(df.head())
    results_adf = adf_test(df)
    results_kpss = kpss_test(df)
    # results_acf_pacf = acf_pacf(df)
    results_pp = pp_test(df)
    
    print("Results obtained")
    
    # guardar resultados en un archivo de texto 
    
    results = f"ADF Test:\nADF Statistic: {results_adf[0]}\nKPSS Test: {results_kpss}\nPP Test: {results_pp}"
    
    print(f"Results ADF: ")
    print(results_adf)
    print(f"-----------------------------------------")

    print(f"Results KPSS: ")
    print(results_kpss)
    print(f"-----------------------------------------")
    
    print(f"Results PP: ")
    print(results_pp)

    
    write_results_to_txt(results, file_save)
    
    print("Results saved")
    
if __name__ == '__main__':
    main()
    