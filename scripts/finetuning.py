import pandas as pd
import numpy as np
import random
import os
import sys
sys.path.append('..')
import argparse
from multiprocessing import Process
from modules.constants import constants
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')




if __name__== "__main__":
    parser = argparse.ArgumentParser(description = 'Parameters for dqn model')
    parser.add_argument('-s', '--seed', help='Seed to use in experiment', type=int, default=42)
    parser.add_argument('-t', '--steps', help='Number of timesteps', type=int, default=int(5e7))
    parser.add_argument('-m', '--metric', help='best or avg', type=str, default='f1')
    args = parser.parse_args()
    constants.init(args)

    from modules import utils

    random.seed(constants.SEED)
    np.random.seed(constants.SEED)
    os.environ['PYTHONHASHSEED']=str(constants.SEED)
    tf.set_random_seed(constants.SEED)
    tf.compat.v1.set_random_seed(constants.SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    print(f'Seed being used: {constants.SEED}')
    print(f'Number of steps: {args.steps}')

    try:
        train_df = pd.read_csv('../data/rwd_dataset/train_set.csv') 
    except:
        print("Dataset not publicly available. Swith path to synthetic dataset to test script.")
    train_df = train_df.fillna(-1)

    X_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]
    X_train, y_train = np.array(X_train), np.array(y_train) 

   
    model_path = f"../models/synthetic_trained_model.zip.zip"
    parent_dir = f"../models/finetuned"
    utils.train_dqn_model(args.steps, model_path, X_train, y_train, constants.SEED, parent_dir, "finetuned")


    print('DONE!')

   

    

   