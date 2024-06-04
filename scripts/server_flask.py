import json
import keras
import numpy as np
from flask import Flask, request, jsonify
import os
import struct
import redis
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout,Conv1D, MaxPooling1D, Flatten,LSTM # type: ignore
from model_creators import Lstm_model, Conv1D_model, Dense_model
import threading
import pandas as pd
import requests
import click
from flask_cors import CORS # type: ignore


# Inicializar la conexión a Redis
r = redis.Redis(host='localhost', port=6379, db=0)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
model = None
    
def load_best_model (num_outputs = 5):
    
    # checkpoint_filepath_lstm = f'./weights/model_lstm_modelversion_{model_version}_outputs_{n_outputs}_{{loss:.4f}}.weights.h5'
    
    best_model_path = ""
    current_loss = 1000000
    global model
    paths_outputs = []
    
    for file in os.listdir("..\weights"):
        if file.split("_")[5] == str(num_outputs):
            paths_outputs.append(file)
        
    for file in paths_outputs:
        loss = file.split("_")[6].split(".weights")[0]
        if float(loss) < current_loss:
            current_loss = float(loss)
            best_model_path = os.path.join("..\weights", file)
    
    print(best_model_path)
    # print("------------------------------------")
    
    if best_model_path == "":
        raise ValueError(f"Model file not found")
    
    # print(tf.__version__)
    # print(keras.__version__)
    best_model_type = best_model_path.split("_")[1]
    if  best_model_type== "lstm":
        model = Lstm_model(num_outputs)
    elif best_model_type == "conv":
        model = Conv1D_model(num_outputs).model
    elif best_model_type == "dense":
        model = Dense_model(num_outputs)
    else:
        raise ValueError(f"Unrecognized model type in file name: {best_model_path}")
    
    model.load_weights(best_model_path)
    
    # print de los pesos del modelo
    # print(model.model.get_weights())  
            
    return model    
    

def process_task_queue():
    while True:
        task_id = r.lpop('task_queue')
        if task_id is None:
            break
        
        task_id = task_id.decode('utf-8')
        data_bytes = r.get(f'task:{task_id}')
        
        print(f"Processing task {task_id}")
        
        if data_bytes is not None:
            print(f"Task {task_id} started")
            # Convertir los datos binarios a un array de floats
            data = np.frombuffer(data_bytes, dtype=np.float64)
            
            print (f"Data: {data}")
            
            # Reshape the data to the expected input shape of the model
            data = data.reshape(-1, 32, 1)
            
            preds = model.predict(data)
            
            print(f"Task {task_id} completed")
            # print(f"Data: {data}")
            print(f"Predictions: {preds}")
            # Predictions: [[0.92251194 0.92104053 0.922166   0.9214076  0.92190003]]
            
            # Convertir las predicciones a un formato JSON
            preds = preds.tolist()
            result = {"task_id": task_id, "predictions": preds}
            r.set(f'result:{task_id}', json.dumps(result))
            print(f"Result for task {task_id} saved")
        else:
            print(f"Task {task_id} has no data")
            result = {"task_id": task_id, "error": "Task has no data"}
            r.set(f'result:{task_id}', json.dumps(result))
            print(f"Result for task {task_id} saved")

@app.route("/")
def general():    
    return "Api Rest para predicción de series temporales"


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    
    # print("Data received: ", request.get_json()) 

    if request.method == "POST":
        request_data = request.get_json()
        if request_data and request_data.get("data"):
            data_predict = np.array(request_data["data"])
            
            # print(f"Data type: {type(data)}")

            # print(f"Data received: {data}")

            # Agregar la tarea a la cola de Redis
            task_id = r.incr('task_id')
            r.rpush('task_queue', task_id)
            r.set(f'task:{task_id}', data_predict.tobytes())
            
            # print (f"Task ID key and data: {task_id}, {r.get(f'task:{task_id}')}")
            
            # print(f"Task {task_id} added to the queue")

            data["task_id"] = task_id
            data["success"] = True

            # Procesar la tarea de la cola
            process_task_queue()

    return jsonify(data)

@app.route("/predict/results/<task_id>", methods=["GET"])
def get_results(task_id):
    
    result = r.get(f'result:{task_id}')
    
    print(f"Result for task {task_id}: {result}")
    if result is not None:
        result = json.loads(result)
    else:
        result = {"error": "Task not found"}
    return jsonify(result)

def run_task_queue_worker():
    while True:
        process_task_queue()
        
@click.command()
@click.option('--num-output', type=int, default=5, help="Number of outputs")
# python .\server_ideal.py --num-output 5

def main(num_output : int) -> None:
    print("* Loading Keras model and Flask starting server...")
    print("Please wait until the server has fully started")
    load_best_model(num_output)
    # Iniciar un hilo separado para procesar la cola de tareas
    worker_thread = threading.Thread(target=run_task_queue_worker)
    worker_thread.start()
    
    app.run()

if __name__ == "__main__":
    main()