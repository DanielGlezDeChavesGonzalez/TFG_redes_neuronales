# Descripcion: Script que contiene el servidor de Flask que se encarga de recibir los datos de entrada, procesarlos y devolver la predicción del modelo LSTM.

import numpy as np
from flask import Flask, request, jsonify
import os
import redis
import tensorflow as tf

# initialize our Flask application and the Keras model
app = Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None

def load_best_model ():
    
    # checkpoint_filepath_lstm = f'./weights/model_lstm_{model_version}_dataset_{dataset_version}_{{loss:.5f}}.weights.h5'
    
    global model
    
    best_model_path = ""
    current_loss = 1000000
    
    for root, files in os.walk("./weights"):
        for file in files:
            if file.endswith(".h5"):
                loss = file.split("_")[-1].split(".")[0]
                if float(loss) < current_loss:
                    current_loss = float(loss)
                    best_model_path = os.path.join(root, file)
    
    print(best_model_path)
    
    if best_model_path:
        best_model = tf.keras.models.load_model(best_model_path)
        model = best_model
        return best_model    
    else:
        return None

def prepare_data(data):
    data = np.frombuffer(data, dtype=np.float32)
    data = data.reshape(1, 32)
    return data

@app.route("/")
def general():
    return "Welcome to the LSTM model API using as a queue Redis"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

	# ensure an array was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("data"):
            data = request.files["data"].read()
            data = prepare_data(data)
            preds = model.predict(data)
            data["predictions"] = []
            
            for pred in preds:
                r = {"value": pred}
                data["predictions"].append(r)
            
            data["success"] = True
                
            
	# return the data dictionary as a JSON response
    return jsonify(data)

# # if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..." 
           "please wait until server has fully started"))
    load_best_model()
    app.run()