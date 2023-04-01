from fsspec import Callback
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError

import pandas as pd
import logging

logger = logging.getLogger(__name__)

class NNRegressor():
    """
        Create a Neural Network Regressor Model with a default architecture. 
        It has 1 input layer (with 256 neurons), one hidden layer (with 256 neurons) and 1 output layer.
        
    """
    
    def __init__(self):
        logger.info("Initializing trainer ...")
        self._model = self._create_model()
        self._trained_model = None
    
    def _create_model(self):
        """
        Create a tf.keras model and compile model

        Returns:
            tf.keras.Model : NN compiled model
        """
        
        model = Sequential([
            Dense(256, activation = 'tanh'),
            Dense(256, activation = 'tanh'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        
        return model
    
    def fit(self, X_train: pd.DataFrame, y_train : pd.DataFrame,  batch_size = 256, epochs = 15, validation_split = 0.1):
        """
        
        Function to train neural network model.

        Args:
            X_train (pd.DataFrame): data features to be used for training.
            y_train (pd.DataFrame): data target to be used for training.
            batch_size (int, optional): Batch size to use in training. Defaults to 256.
            epochs (int, optional): Number of epochs to train for. Defaults to 15.
            validation_split (float, optional): Ratio of data to take as validation set for evaluation validation loss after each epochs. Defaults to 0.1.
        """
        
        self._trained_model_history  = self._model.fit(X_train, y_train, 
                                                           epochs = epochs, batch_size = batch_size, 
                                                           validation_split = validation_split,
                                                           verbose = 0,
                                                           callbacks = [LoggingCallback(logger.info)]
                                                           )
        
    
    def predict(self, X_test : pd.DataFrame) -> list:
        """
        Function to use trained model for prediction.

        Args:
            X_test (pd.DataFrame): data features to be used for training.

        Returns:
            list: model predictions.
        """
        
        preds = self._model.predict(X_test, verbose = 0)
        
        return preds
    
    def save_model(self, path : str = None):
        """
        Function to save trained model .h5 file.

        Args:
            path (str, optional): specific path to store model on. Defaults to None.
        """

        self._model.save('model_1.h5', )
        
        
class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """

    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):

        msg = "Epoch: %i %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)