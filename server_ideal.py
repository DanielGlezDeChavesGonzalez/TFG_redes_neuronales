import json
import keras
import numpy as np
from flask import Flask, request, jsonify
import os
import redis
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout,Conv1D, MaxPooling1D, Flatten,LSTM # type: ignore
from model_creators import Lstm_model, Conv1D_model, Dense_model

# Inicializar la conexión a Redis
r = redis.Redis(host='localhost', port=6379, db=0)
app = Flask(__name__)
model = None

def load_best_model ():
    
    # checkpoint_filepath_lstm = f'./weights/model_lstm_{model_version}_dataset_{dataset_version}_{{loss:.5f}}.weights.h5'
    
    best_model_path = ""
    current_loss = 1000000
    
    global model
    
    for file in os.listdir(".\weights"):
        loss = file.split("_")[5].split(".weights")[0]
        if float(loss) < current_loss:
            current_loss = float(loss)
            best_model_path = os.path.join(".\weights", file)
    
    print(best_model_path)
    print("------------------------------------")
    
    # print(tf.__version__)
    # print(keras.__version__)
    if best_model_path.split("_")[1] == "lstm":
        model = Lstm_model()
    elif best_model_path.split("_")[1] == "conv1d":
        model = Conv1D_model()
    elif best_model_path.split("_")[1] == "dense":
        model = Dense_model()
    else:
        raise ValueError(f"Unrecognized model type in file name: {best_model_path}")
    
    model.load_weights(best_model_path)
        
    return model    
    
def prepare_data(data):
    # image = Image.open(io.BytesIO(data))
    # image = preprocess_image(image)
    # data = image_to_array(image)
    
    # data = preprocess_text(data)
    
    data = np.frombuffer(data, dtype=np.float32)
    data = data.reshape(1, 32)  # Ajustar la forma según sea necesario
    
    return data

def process_task_queue():
    while True:
        task_id = r.lpop('task_queue')
        if task_id is None:
            break
        
        task_id = task_id.decode('utf-8')
        data_bytes = r.get(f'task:{task_id}')
        
        if data_bytes is not None:
            data = prepare_data(data_bytes)
            preds = model.predict(data)
            
            predictions = []
            for pred in preds:
                predictions.append({"value": float(pred)})
            
            result = {"predictions": predictions}
            r.set(f'result:{task_id}', json.dumps(result))

@app.route("/")
def general():
    return "Welcome to the LSTM model API using as a queue Redis"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    
    if request.method == "POST":
        if request.files.get("data"):
            data_bytes = request.files["data"].read()
            
            # Agregar la tarea a la cola de Redis
            task_id = r.incr('task_id')
            r.rpush('task_queue', task_id)
            r.set(f'task:{task_id}', data_bytes)
            
            data["task_id"] = task_id
            data["success"] = True
            
            # Procesar la tarea de la cola
            process_task_queue()
    
    return jsonify(data)

@app.route("/results/<task_id>", methods=["GET"])
def get_results(task_id):
    result = r.get(f'result:{task_id}')
    if result is None:
        return jsonify({"error": "Task not found or not completed yet"}), 404
    
    return jsonify(json.loads(result))

import threading

def run_task_queue_worker():
    while True:
        process_task_queue()

if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server...")
    print("Please wait until the server has fully started")
    load_best_model()
    
    # Iniciar un hilo separado para procesar la cola de tareas
    worker_thread = threading.Thread(target=run_task_queue_worker)
    worker_thread.start()
    
    app.run()