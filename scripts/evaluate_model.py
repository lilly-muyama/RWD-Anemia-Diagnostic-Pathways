import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import random
import os
import sys
sys.path.append('..')
from modules.constants import constants
from multiprocessing import Process

def test_dqn_model(model_name, seed, steps, X_test, y_test):
    path = f'../models/{model_name}_model.zip'

    
    perf_arr = []
    
    model = utils.load_dqn(path)
    test_df = utils.evaluate_dqn(model, X_test, y_test)
    acc, f1, roc_auc = utils.test(test_df.y_actual, test_df.y_pred)

    perf_dict = {'model_metric':name, 'seed':seed, 'acc':acc, 'f1':f1, 'roc-auc':roc_auc, 'avg_length':test_df.episode_length.mean(), 'min_length':test_df.episode_length.min(), 'max_length':test_df.episode_length.max()}
    print(f'{model_name}: {perf_dict}')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Parameters for script')
    parser.add_argument('-m', '--model_name', help='Name of model. Should be one of "finetuned", "rwd_trained", or "synthetic_trained".', type=str, default="finetuned")
    
   
    args = parser.parse_args()
    constants.init(args)

    from modules import utils

    SEED = constants.SEED
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED']=str(SEED)
    tf.set_random_seed(SEED)
    tf.compat.v1.set_random_seed(SEED)

    try:
        test_data = pd.read_csv('../data/rwd_dataset/test_set.csv')
    except:
        print("Dataset not publicly available. Swith path to synthetic dataset to test script.")

    test_data = test_data.fillna(-1)
    X_test = test_data.iloc[:, 0:-1]
    y_test = test_data.iloc[:, -1]
    X_test, y_test = np.array(X_test), np.array(y_test)

    test_dqn_model(args.model_name, constants.SEED, args.steps, X_test, y_test)

    print('DONE!')