# Descripción: Enviar una solicitud POST al servidor REST API para realizar una predicción con un modelo de Keras.
 
import requests
import numpy as np

KERAS_REST_API_URL = "http://localhost:5000/predict"
DATA_PATH = "./datos_peticion.txt"  # Reemplazar con la ruta del archivo de datos

# Cargar los datos de entrada desde un archivo binario
with open(DATA_PATH, "rb") as f:
    data = f.read()

payload = {"data": data}

print("Enviando solicitud POST al servidor REST API...")
print("Payload: ", payload)


# Enviar la solicitud POST
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# Comprobar si la solicitud fue exitosa
if r["success"]:
    task_id = r["task_id"]
    print(f"Tarea enviada correctamente. ID de tarea: {task_id}")
    
    # Obtener los resultados de la predicción
    results_url = f"{KERAS_REST_API_URL}/results/{task_id}"
    results = requests.get(results_url).json()
    
    # Imprimir las predicciones
    if "predictions" in results:
        predictions = results["predictions"]
        for i, pred in enumerate(predictions):
            print(f"{i+1}. Predicción: {pred['value']:.4f}")
    else:
        print("Error al obtener los resultados:", results)
else:
    print("Error al enviar la solicitud:", r)