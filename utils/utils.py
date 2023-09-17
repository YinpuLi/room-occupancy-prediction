import pandas as pd
from scipy.io import loadmat
from datetime import datetime, timedelta
import joblib
import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)



from utils.constants import *



def get_absolute_path(
          file_name:str='Occupancy_Estimation.csv'
          , rel_path:str='data'
        #   , base_dir: str = os.path.abspath(os.path.dirname(__file__))
          , base_dir = BASE_DIR#os.path.abspath(os.path.join(os.getcwd(), '..'))
):
     return os.path.join(base_dir, rel_path, file_name)



def load_data(file_path):
    loaded_data = loadmat(file_path)
    return loaded_data
    



########## input: for training set 


def save_csv(df, file_path, index=False):
    _dir = os.path.dirname(file_path)
    os.makedirs(_dir, exist_ok=True)

    df.to_csv(file_path, index=index)






def record_running_time(start_time):
    end_time = datetime.now()
    running_time = end_time - start_time
    return running_time

def save_model(file_name, model, model_info=None):
    """
    Save a model and its associated info (if provided) to a file using joblib.

    Parameters:
    file_name (str): The name of the file to save the model.
    model (object): The model object to be saved.
    model_info (dict, optional): Additional information about the model (default: None).
    """
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    # elif os.path.exists(file_name):
    #     raise ValueError(f"File '{file_name}' already exists. Please provide a new file name.")

    joblib.dump((model, model_info), file_name)



def load_model(file_name):
    """
    Load a saved model and its associated info using joblib.

    Parameters:
    file_name (str): The name of the file containing the saved model.

    Returns:
    model: The loaded model object.
    model_info: Additional information about the model (if available).
    
    # Load the model and its info
    loaded_model, loaded_model_info = load_model(best_xgb_file)
    """
    return joblib.load(file_name)


