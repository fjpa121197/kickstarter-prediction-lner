import argparse
import sys
import datetime
import traceback
import os
import logging

# Gets or creates a logger
logger = logging.getLogger(__name__)  

# set log level
logger.setLevel(logging.DEBUG)

# define file handler and set formatter
file_handler = logging.FileHandler('logfile.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)


def main():
    
    logger.info("Initiating ML pipeline run ...")
    
    parser = argparse.ArgumentParser(
        description="ML Pipeline, triggers a ML pipeline that trains a neural network based on provided a data file path."
    )
    
    parser.add_argument(
        "-d",
        "--data-file",
        dest="data_file_path",
        type=str,
        required=False,
        help="""Data to be used in the pipeline, must be a path.""",
    )
    
    # Get arguments passed when invoking option
    # As for know, it only takes a data file path argument (which must be locally)
    args = parser.parse_args()


    # TODO: Check file path exists and there is a data file to work with before going into try/except block. This can be a function in utils module.

    
    try:
        
        # TODO: Call pipeline.preprocess so it preprocesses data (limited to separation of features/target and train_test_split)
        
        # TODO: Call pipeline.train so it initializes model with the necessary configurations and ready to be trained
        
        # TODO: Call pipeline.train so it trains model
        
        # TODO: Call pipeline.eval so it uses trained model to predict and evaluate predictions based on test set generated in preprocessing stage
        
        # TODO: Call pipeline.train so it saves trained model to disk
        
        print("Hi")
    except Exception as e:
        print(f"Exeception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()