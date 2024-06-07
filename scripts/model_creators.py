from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout,Conv1D, MaxPooling1D, Flatten,LSTM , Lambda # type: ignore
import tensorflow as tf

class Lstm_model :
    
    def __init__(self, n_outputs):
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(32,1)),
            # Dropout(0.2),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=True),
            # Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(20),
            Dense(n_outputs)
        ])
        
    def load_weights(self, path):
        self.model.load_weights(path)
        
    def predict (self, data):
        return self.model.predict(data)
    
    def train_on_batch (self, data, target):
        return self.model.train_on_batch(data, target)
    
    def compile(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss)
        
class Conv1D_model:
    
    def __init__(self, n_outputs):
        self.model = Sequential([
            Conv1D(32, 2, activation='relu', input_shape=(32,1)),
            Conv1D(32, 2, activation='relu'),
            MaxPooling1D(2),
            Conv1D(64, 2, activation='relu'),
            Conv1D(64, 2, activation='relu'),
            MaxPooling1D(2),
            Conv1D(128, 2, activation='relu'),
            Conv1D(32, 2, activation='relu'),
            # MaxPooling1D(2),
            Conv1D(200, 2, activation='relu'),
            # MaxPooling1D(2),
            Flatten(),
            Dense(100),
            Dense(n_outputs)
        ])
        
    def load_weights(self, path):
        self.model.load_weights(path)
        
    def predict (self, data):
        return self.model.predict(data)
    
    def train_on_batch (self, data, target):
        return self.model.train_on_batch(data, target)
    
    def compile(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss)
class Dense_model:
        
    def __init__(self, n_outputs):
        self.model = Sequential([
            Dense(32, activation='relu', input_shape=(32,1)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(100),
            Dense(n_outputs)
        ])
        
    def load_weights(self, path):
        self.model.load_weights(path)
        
    def predict (self, data):
        return self.model.predict(data)
    
    def train_on_batch (self, data, target):
        return self.model.train_on_batch(data, target)
    
    def compile(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss)
        
    