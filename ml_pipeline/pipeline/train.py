import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError


class NNRegressor():
    
    def __init__(self):
        print("Initializing trainer ...")
        self._model = self._create_model()
        self._trained_model = None
    
    def _create_model(self):
        
        model = Sequential([
            Dense(256, activation = 'tanh'),
            Dense(256, activation = 'tanh'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        
        return model
    
    def fit(self, X_train, y_train,  batch_size = 256, epochs = 15, validation_split = 0.1):
        
        self._trained_model_history  = self._model.fit(X_train, y_train, 
                                                           epochs = epochs, batch_size = batch_size, 
                                                           validation_split = validation_split,
                                                           verbose = 2
                                                           )
        
    
    def predict(self, X_test):
        
        preds = self._model.predict(X_test, verbose = 2)
        
        return preds
    
    def save_model(self, path = None):

        self._model.save('model_1.h5')