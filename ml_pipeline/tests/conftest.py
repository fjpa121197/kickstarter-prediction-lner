import pytest
import os
from pipeline.preprocess import Preprocessor
from pipeline.train import NNRegressor

sample_data_file_path = 'tests/sample-data/kickstarter_train_test_onehot.csv'
target = 'usd_smooth_pledged_per_day'
index_col = 'ID'

if os.path.isfile('model_1.h5'):
    print("removing model file ...")
    os.remove('model_1.h5')
    
else:
    print("No model artifacts found ... skipping deleting")

@pytest.fixture()
def preprocessor():
    return Preprocessor(data_file_path= sample_data_file_path, target=target, random_state= 212, index_col= index_col)



@pytest.fixture()
def fit_preprocessor():
    
    X_train, X_test, y_train, y_test = Preprocessor(data_file_path= sample_data_file_path, target=target, 
                                                    random_state= 212, index_col= index_col).split_dataset(test_size = 200)
    
    return X_train, X_test, y_train, y_test

@pytest.fixture()
def trainer():
    return NNRegressor()