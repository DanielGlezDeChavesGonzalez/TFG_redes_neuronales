import click
import os
import tensorflow as tf


def load_best_model ():
    
    # checkpoint_filepath_lstm = f'./weights/model_lstm_{model_version}_dataset_{dataset_version}_{{loss:.5f}}.weights.h5'
    
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
        return best_model
    else:
        return None
    
@click.command()
@click.option("--data-received", type=str, default=".", help="Data to use for prediction.")

def main(data_received : str) -> None:
    
    best_model = load_best_model()
    
    data_received = data_received.split(",")
    
    data_received = [float(i) for i in data_received]
    
    data_received = tf.convert_to_tensor(data_received)
    
    data_received = tf.reshape(data_received, (1, len(data_received), 1))
    
    prediction = best_model.predict(data_received)
    
    print(f"Prediction: {prediction}")
    
if __name__ == "__main__":
    main()
    
    