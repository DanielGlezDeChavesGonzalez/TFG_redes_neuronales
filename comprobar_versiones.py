import tensorflow as tf

# Verificar la versión de TensorFlow
print("TensorFlow version:", tf.__version__)

# Listar los dispositivos disponibles
physical_devices = tf.config.list_physical_devices()
print("Physical devices:", physical_devices)

# Listar los dispositivos GPU disponibles
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)

# Verificar si TensorFlow está usando la GPU con DirectML
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"Logical GPUs: {logical_gpus}")
        except RuntimeError as e:
            print(e)
