import pandas as pd
import numpy as np
import random 
import os
import sys
sys.path.append('..')
import tensorflow as tf
from modules.constants import constants
import argparse
from multiprocessing import Process

def run_dqn_model(model_type, seed, steps):
    dir_name = f'seed_{seed}_{steps}'
    parent_dir = f'../models/synthetic/{model_type}' # to change
    path = os.path.join(parent_dir, dir_name)
    os.mkdir(path)
  
    if model_type=='dqn':
        model = utils.stable_vanilla_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dqn', filename=f'dqn_{steps}')
    elif model_type=='ddqn':
        model = utils.stable_double_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='ddqn', filename=f'ddqn_{steps}')
    elif model_type== 'dueling_dqn':
        model = utils.stable_dueling_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dueling_dqn', filename=f'dueling_dqn_{steps}')
    elif model_type == 'dueling_ddqn':
        model = utils.stable_dueling_ddqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dueling_ddqn', filename=f'dueling_ddqn_{steps}')
    elif model_type =='dqn_per':
        model = utils.stable_vanilla_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dqn_per', filename=f'dqn_per_{steps}', per=True)
    elif model_type == 'ddqn_per':
        model = utils.stable_double_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='ddqn_per', filename=f'ddqn_per_{steps}', per=True)
    elif model_type == 'dueling_dqn_per':
        model = utils.stable_dueling_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dueling_dqn_per', filename=f'dueling_dqn_per_{steps}', per=True)
    elif model_type == 'dueling_ddqn_per':
        model = utils.stable_dueling_ddqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dueling_ddqn_per', filename=f'dueling_ddqn_per_{steps}', per=True)
    else:
        raise ValueError('Unknown model type!')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Parameters for dqn model')
    parser.add_argument('-t', '--steps', help='Number of timesteps to train the model', type=int, default=int(50e6))
    parser.add_argument('-s', '--seed', help='Seed to use in experiment', type=int)
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



    train_df = pd.read_csv('../data/synthetic_dataset/new_new/train_set.csv') # to change
    # train_df = utils.remove_no_anems(train_df, 1/2)
    train_df = train_df.fillna(-1)

    X_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]
    X_train, y_train = np.array(X_train), np.array(y_train)   



    model_names = ['dueling_dqn_per', 'dueling_ddqn_per']
    procs = []

    for name in model_names:
        proc = Process(target=run_dqn_model, args=(name, constants.SEED, args.steps))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


    print('All jobs completed')




    
   
