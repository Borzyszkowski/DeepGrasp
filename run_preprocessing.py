"""
This code will process data and save the .pt files in the out_path folder.
Next, you can directly use the dataloader to load and use the data.    
"""
import argparse
import logging
import os

from tools.utils import makelogger, INTENTS, OBJECTS, SUBJECTS
from tools.cfg_parser import Config
from core.data_preparation.data_preprocessing import DataSet


def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description='DeepGrasp-data-extraction')
    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--out-path', required=True, type=str,
                        help='The path to the folder to save the processed data')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing SMPL-X models')
    parser.add_argument('--process-id', required=False, default='DATA_V04', type=str,
                        help='The appropriate ID for the processed data (folder name)')
    return parser.parse_args()


if __name__ == '__main__':
    instructions = ''' 
    Please do the following steps before starting the data extraction:
    1. Download GRAB dataset from the website https://grab.is.tue.mpg.de/ 
    2. Set the grab_path, out_path and model_path to the correct folders
    3. Change the configuration file for your desired data
       WARNING: saving vertices requires a high-capacity RAM memory.
    '''
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parse_args()
    cwd = os.getcwd()

    remove_objects = []     # FILL_IN: Specify objects to remove from the dataset
    remove_subjects = []    # FILL_IN: Specify subjects to remove from the dataset
    keep_intents = ['all']  # FILL_IN: Specify which intents to keep in the dataset
    
    assert all(item in OBJECTS for item in remove_objects), f"Invalid value in remove_objects: {remove_objects}." 
    assert all(item in SUBJECTS for item in remove_subjects), f"Invalid value in remove_subjects: {remove_subjects}." 
    assert all(item in INTENTS for item in keep_intents), f"Invalid value in keep_intents: {keep_intents}." 

    # Define the experiment configuration (user config overwrites default)
    user_cfg_path = os.path.join(cwd, 'configs/preprocessing_cfg.yml')
    default_config = {
        'grab_path': args.grab_path,
        'model_path': args.model_path,
        'out_path': os.path.join(args.out_path, args.process_id),
        'keep_intents': keep_intents,        # keep these intents in the dataset
        'remove_objects': remove_objects,    # remove these objects from the dataset
        'remove_subjects': remove_subjects,  # remove these subjects from the dataset
    }

    config = Config(default_config, user_cfg_path)
    config.write_cfg(os.path.join(config.out_path, 'data_preprocessing_cfg.yaml'))
    makelogger(os.path.join(config.out_path, 'data_preprocessing.log'), mode='a')

    logging.info(instructions)
    DataSet(config)
