import pytest
import pandas as pd



def test_preprocessor_init(preprocessor):
    
    # Check required parameters are initialized correctly and valid values are given
    assert preprocessor.data_file_path is not None
    assert len(preprocessor.target) > 0
    assert preprocessor.random_state >= 0 and preprocessor.random_state <= (2**32 -1)
    if preprocessor.index_col:
        assert len(preprocessor.index_col) > 0
        
    
def test_preprocessor_df_create(preprocessor):
     
    # Check loaded file produced a dataframe with desired format
    assert isinstance(preprocessor.data_df, pd.DataFrame)
    assert not preprocessor.data_df.empty
    if preprocessor.index_col:
        assert preprocessor.data_df.index.name == preprocessor.index_col
    
    # Check target column is in dataframe
    assert preprocessor.target in preprocessor.data_df.columns
        
        
def test_preprocessor_split_dataset(preprocessor, test_size = 400):
    
    # Check drop of rows with missing values in target is correct
    X_train, X_test, y_train, y_test = preprocessor.split_dataset(test_size = test_size)
    
    assert y_train.isnull().sum() == 0
    assert y_test.isnull().sum() == 0
    
    # Check split of data corresponds to the size given by test_size parameter
    assert X_test.shape[0] == test_size
    assert X_train.shape[1] == X_test.shape[1]
    assert X_test.shape[0] == y_test.shape[0]
    
    # Check separation of features and target is done correctly
    assert preprocessor.target not in X_train.columns
