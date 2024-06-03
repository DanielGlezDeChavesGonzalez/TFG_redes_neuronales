from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout,Conv1D, MaxPooling1D, Flatten,LSTM # type: ignore


class Lstm_model :
    
    def __init__(self, n_outputs=10):
        self.model = Sequential([
            LSTM(100, input_shape=(32,1), return_sequences=True),
            Dropout(0.2),
            LSTM(200, return_sequences=True),
            LSTM(200, return_sequences=True),
            LSTM(200, return_sequences=True),
            LSTM(400, return_sequences=True),
            Dropout(0.2),
            LSTM(300, return_sequences=True),
            Dropout(0.2),
            Dense(100),
            Dense(n_outputs)
        ])
        
    def load_weights(self, path):
        self.model.load_weights(path)
        
    def predict (self, data):
        return self.model.predict(data)

class Conv1D_model:
    
    def __init__(self, n_outputs):
        self.model = Sequential([
            Conv1D(100, 2, activation='relu', input_shape=(32,1)),
            Conv1D(100, 2, activation='relu'),
            MaxPooling1D(2),
            Conv1D(200, 2, activation='relu'),
            Conv1D(200, 2, activation='relu'),
            Conv1D(400, 2, activation='relu'),
            MaxPooling1D(2),
            Conv1D(300, 2, activation='relu'),
            Conv1D(300, 2, activation='relu'),
            # MaxPooling1D(2),
            Conv1D(200, 2, activation='relu'),
            Conv1D(200, 2, activation='relu'),
            # MaxPooling1D(2),
            Flatten(),
            Dense(200),
            Dense(100),
            Dense(n_outputs)
        ])
        
    def load_weights(self, path):
        self.model.load_weights(path)
        
    def predict (self, data):
        return self.model.predict(data)

class Dense_model:
        
    def __init__(self, n_outputs):
        self.model = Sequential([
            Dense(100, activation='relu', input_shape=(32,1)),
            Dropout(0.2),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dropout(0.2),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(100),
            Dense(n_outputs)
        ])
        
    def load_weights(self, path):
        self.model.load_weights(path)
        
    def predict (self, data):
        return self.model.predict(data)
    
    
        
    