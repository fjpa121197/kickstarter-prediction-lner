import argparse
import sys
import traceback
import os
import logging


from pipeline.preprocess import Preprocessor
from pipeline.train import NNRegressor


def main():
    
    logging.basicConfig(filename='mlpipeline.log', level=logging.DEBUG, format='%(asctime)s %(name)s: %(message)s')
    logging.info("Initiating ML pipeline run ...")
    
    parser = argparse.ArgumentParser(
        description="ML Pipeline, triggers a ML pipeline that trains a neural network based on provided a data file path."
    )
    parser.add_argument(
        "-d",
        "--data-file",
        dest="data_file_path",
        type=str,
        required=True,
        help="""Data to be used in the pipeline, must be a path.""",
    )
    
    # Get arguments passed when invoking option
    # As for know, it only takes a data file path argument (which must be locally)
    args = parser.parse_args()
    
    # Check file exists in path
    if os.path.isfile(args.data_file_path):
        data_file_path = args.data_file_path
        
    else:
        logging.error("File path is not valid. Please check directory and file exists.")
        sys.exit(1)


    # Start with pipeline
    try: 
        # Preprocessing Step
        preprocessor = Preprocessor(data_file_path=data_file_path, target = 'usd_smooth_pledged_per_day', 
                                    random_state= 212, index_col='ID')
        X_train, X_test, y_train, y_test = preprocessor.split_dataset(test_size = 64000) # split dataset into train and test set
        
        # Training Step
        trainer = NNRegressor() # Initialize NN model using default architecture
        trainer.fit(X_train, y_train, epochs=15) # Train model using training set and for specific number of epochs
        preds = trainer.predict(X_test) # Get predictions using trained model
        
        # Save artifacts Step
        trainer.save_model()
        
        # end pipeline
        
        logging.info("ML pipeline run finished")
        
    except Exception as e:
        print(f"Exeception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()