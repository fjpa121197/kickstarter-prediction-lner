from sklearn.model_selection import train_test_split
import pandas as pd

class Preprocessor():
    """
    Create a Preprocessor object that handles data preparation steps needed to transform and prepare data for next steps in a ML cycle.
    It takes a dataframe as input, and it handles data loading automatically.
    
    Parameters
    ----------
    
    data_file_path : path of file to be as data (str).
    
    target : name of the column in data to be used as target (str).
    
    random_state : number to be used as seed to get reproducible results (int).
    
    index_col : name of the column to be used in data as index (str - optional).
    
    """
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
        """
        Load csv data file into a dataframe (given that path is correct). For internal use only.

        Args:
            path (str): csv data file path

        Returns:
            df: DataFrame
        """
        
        df = pd.read_csv(path, index_col=self.index_col)
        
        return df
    
    
    def _separate_features_target(self, df):
        """
        Given a dataframe, separates features and target. For internal use only.

        Args:
            df (pd.DataFrame): dataframe to be separated

        Returns:
            X: dataframe only containing features
            y: dataframe only containing target
        """
        
        X = df.drop(self.target, axis = 1)
        y = df[self.target]
        
        return X, y
    
    def split_dataset(self, test_size):
        """
        Separate dataframe into train and test set. Given object was initialized correctly, it will separate data into 2 sets, 
        train and test set, based on test_size parameter.

        Args:
            test_size (int): Number of samples needed in test set (can be integer or a ratio (0.1, 0.2, 0.3, ...)).

        Returns:
            X_train: training data containing features only.
            X_test: testing data containing features only.
            y_train: training data containing target only.
            y_test: test data containing target only.
        """
        
        df = self.data_df
        df = df.dropna(subset=[self.target])
        
        self.train_df, self.test_df = train_test_split(df, random_state= self.random_state, test_size= test_size)
        
        X_train, y_train = self._separate_features_target(self.train_df)
        X_test, y_test = self._separate_features_target(self.test_df)
        
        return X_train, X_test, y_train, y_test
        