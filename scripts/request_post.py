# Descripción: Enviar una solicitud POST al servidor REST API para realizar una predicción con un modelo de Keras.

import requests
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import streamlit as st
## Execution to work 
# 1. Run the server with the command: python -m streamlit run request_server.py

KERAS_REST_API_URL = "http://localhost:5000/predict"
DATA_PATH = "./datos_peticion.csv"  # Reemplazar con la ruta del archivo de datos

# Cargar los datos de entrada desde un archivo csv
data = pd.read_csv(DATA_PATH, sep=';', names=['Timestamp', 'Value'])
data = np.array(data['Value'])

# Crear el payload de la solicitud POST
payload = {"data": data.tolist()}

print("Datos de entrada: ", data)

print("Enviando solicitud POST al servidor REST API...")
print("Payload: ", payload)

# Enviar la solicitud POST
r = requests.post(KERAS_REST_API_URL, json=payload).content

print("Respuesta del servidor REST API: ", r)

# Comprobar si la solicitud fue exitosa
r = json.loads(r)

if r["success"]:
    task_id = r["task_id"]
    print(f"Tarea enviada correctamente. ID de tarea: {task_id}")

    # Obtener los resultados de la predicción
    results_url = f"{KERAS_REST_API_URL}/results/{task_id}"
    results = requests.get(results_url).json()

    print(results)
    # 'predictions': [[0.9225119352340698, 0.9210405349731445, 0.9221659898757935, 0.9214075803756714, 0.9219000339508057]], 'task_id': '40'}
    print("Resultados de la predicción:")

    # Imprimir las predicciones
    if "predictions" in results:
        predictions = results["predictions"]
        print("Predicciones:")
        for i, pred in enumerate(predictions[0]):
            print(f"Paso {i+1}: {pred}")
            
        # Graficar las predicciones
        plt.figure(figsize=(12, 6))
        plt.plot(data, 'g')
        plt.plot(np.arange(len(data), len(data) + len(predictions[0])), predictions[0], 'b')
        plt.title('Predicciones del modelo')
        plt.xlabel('Índice')
        plt.ylabel('Valor')
        plt.show()

    else:
        print("Error al obtener los resultados:", results)
else:
    print("Error al enviar la solicitud:", r)

# if 'post_response' not in st.session_state:
#     st.session_state.post_response = {}

# if 'get_response' not in st.session_state:
#     st.session_state.get_response = {}

# if 'file' not in st.session_state:
#     st.session_state.file = None

# if 'button_predict' not in st.session_state:
#     st.session_state.button_predict = False
    
# def on_click(value):
#     st.session_state.button_predict = value

# st.session_state.file = st.file_uploader("Upload a csv to predict next steps", type=["csv"])

# if st.session_state.file is not None:
#     data = pd.read_csv(st.session_state.file, sep=';', names=['Timestamp', 'Value'])
#     st.write(data)
#     st.write("Data uploaded")

#     st.button("Predict", on_click=on_click(True), kwargs=dict(value=not st.session_state.button_predict))

#     if st.session_state.button_predict:
#         st.write("Predicting...")
#         st.session_state.post_response = requests.post("http://localhost:5000/predict", json={"data": data['Value'].tolist()})
#         st.write(st.session_state.post_response.json())

#     if st.button("Get results", disabled= not st.session_state.button_predict):
#         st.write(st.session_state.post_response.json())
#         task_id = st.session_state.post_response.json()['task_id']
#         st.session_state.get_response = requests.get(f"http://localhost:5000/predict/results/{task_id}")
#         st.write("Results:")
#         st.write(st.session_state.get_response.json())

# else:
#     st.write("No file uploaded")