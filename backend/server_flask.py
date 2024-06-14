import json
import numpy as np
from flask import Flask, request, jsonify
import redis
import tensorflow as tf
import threading
import pandas as pd
import click
from flask_cors import CORS # type: ignore


# Inicializar la conexión a Redis
r = redis.Redis(host='redis', port=6379, db=0)
# r = redis.Redis(host='localhost', port=6379, db=0)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
# model = tf.keras.Sequential([
#         tf.keras.layers.Conv1D(32, 2, activation='relu', input_shape=(32,1)),
#         tf.keras.layers.Conv1D(32, 2, activation='relu'),
#         tf.keras.layers.MaxPooling1D(2),
#         tf.keras.layers.Conv1D(64, 2, activation='relu'),
#         tf.keras.layers.Conv1D(64, 2, activation='relu'),
#         tf.keras.layers.MaxPooling1D(2),
#         tf.keras.layers.Conv1D(128, 2, activation='relu'),
#         tf.keras.layers.Conv1D(32, 2, activation='relu'),
#         # tf.keras.layers.# MaxPooling1D(2),
#         tf.keras.layers.Conv1D(200, 2, activation='relu'),
#         # tf.keras.layers.# MaxPooling1D(2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(100),
#         tf.keras.layers.Dense(5)
#     ])

model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(32,1)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(20),
            tf.keras.layers.Dense(5)
        ])

# model = tf.keras.Sequential([
#             tf.keras.layers.Flatten(input_shape=(32, 1)),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(5)
#         ])
        

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
            data = data.reshape(1, 32, 1)
            
            preds = model.predict(data)
            
            preds = preds.flatten()
            
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
    path_model = "model_lstmbanc_1.csv_modelversion_1_outputs_5_0.0100342305.weights.h5"
    # load_best_model(num_output)
   
    model.load_weights(path_model)
    # Iniciar un hilo separado para procesar la cola de tareas
    worker_thread = threading.Thread(target=run_task_queue_worker)
    worker_thread.start()
    
    # app.run()
    app.run(host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()