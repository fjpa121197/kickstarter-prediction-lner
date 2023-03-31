from sklearn.model_selection import train_test_split
import pandas as pd

class Preprocessor():
    def __init__(self, data_file_path : str, target : str, random_state: int, index_col : str = None):
        
        print("Initializing data preprocessor ... ")
        
        # Define initial given attributes
        self.data_file_path = data_file_path
        self.target = target
        self.index_col = index_col
        self.random_state = random_state
        
        # Load data file into dataframe
        self.data_df = self._load_file_dataframe(data_file_path)
        
        # Initialize empty objects to store train and test data
        
        self.train_df = None
        self.test_df = None
        
        
    def _load_file_dataframe(self, path : str):
        
        df = pd.read_csv(path, index_col=self.index_col)
        
        return df
    
    
    def _separate_features_target(self, df):
        
        X = df.drop(self.target, axis = 1)
        y = df[self.target]
        
        return X, y
    
    def split_dataset(self, test_size):
        
        df = self.data_df
        df = df.dropna(subset=[self.target])
        
        self.train_df, self.test_df = train_test_split(df, random_state= self.random_state, test_size= test_size)
        
        X_train, y_train = self._separate_features_target(self.train_df)
        X_test, y_test = self._separate_features_target(self.test_df)
        
        return X_train, X_test, y_train, y_test
        