from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout,Conv1D, MaxPooling1D, Flatten,LSTM # type: ignore


class Lstm_model :
    
    def __init__(self):
        self.model = Sequential([
            LSTM(64, input_shape=(32,1), return_sequences=True),
            Dropout(0.2),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(256, return_sequences=False),
            Dropout(0.2),
            Dense(5)
        ])
        

class Conv1D_model:
    
    def __init__(self):
        self.model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=(32,1)),
            MaxPooling1D(3),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(3),
            Conv1D(256, 3, activation='relu'),
            MaxPooling1D(3),
            Flatten(),
            Dense(5)
        ])
        

class Dense_model:
        
    def __init__(self):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(32,1)),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(5)
        ])

    
        
    