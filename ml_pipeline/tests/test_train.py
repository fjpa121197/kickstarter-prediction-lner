import pytest
import tensorflow as tf
import os

def test_model_creation(trainer):
    
    # Check model is created correctly
    assert isinstance(trainer._model, tf.keras.Model)
    
    # Check  model has optimizer, and loss metrics associated to it (happens when model is compiled)
    model_config = trainer._model.get_compile_config()
    assert model_config['optimizer'] is not None
    assert model_config['loss'] is not None
    assert model_config['metrics'] is not None
    
    
def test_model_fit(trainer, fit_preprocessor):
    
    # Use sample preprocessed data for training
    X_train, _, y_train, _ = fit_preprocessor[0], fit_preprocessor[1], fit_preprocessor[2], fit_preprocessor[3]
    
    # Fit model
    trainer.fit(X_train, y_train, epochs = 2)
    # Check fitting of model partially works and values for loss are decreasing
    model_history = trainer._trained_model_history.history
    assert model_history['loss'][0] > model_history['loss'][-1]
    assert model_history['val_loss'][0] > model_history['val_loss'][-1]
    
def test_model_predict(trainer, fit_preprocessor):
    
    # Use sample preprocessed data for predictions
    _, X_test, _, y_test = fit_preprocessor[0], fit_preprocessor[1], fit_preprocessor[2], fit_preprocessor[3]
    
    # Check if model generates predictions correctly (same shape as provided input)
    preds = trainer.predict(X_test) # Get predictions
    assert preds is not None
    
    preds = preds.reshape(-1) # Reshape them so they are in 1-d list

    assert preds.shape[0] == y_test.shape[0]
    assert len(set(preds)) > 1
    
    
def test_save_model(trainer, fit_preprocessor):

    # Use sample preprocessed data for training
    X_train, _, y_train, _ = fit_preprocessor[0], fit_preprocessor[1], fit_preprocessor[2], fit_preprocessor[3]
    
    # Fit model
    trainer.fit(X_train, y_train, epochs = 2)
    
    trainer.save_model()
    
    assert os.path.isfile('model_1.h5')
    
    