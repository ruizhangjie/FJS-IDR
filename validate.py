import copy
import os
import time

import torch

from PPO_Model.PPO import Memory
from env.DREFJSP_Env import DREFJSP
from params import configs


def get_validate_env():
    '''
    Generate and return the validation environment from the validation set ()
    '''
    file_path = "./data_dev/{0}{1}/".format(str.zfill(str(configs.num_jobs), 2), str.zfill(str(configs.num_mas), 2))
    batch_size = configs.valid_batch_size
    valid_data_files = os.listdir(file_path)
    sorted_file_names = sorted(valid_data_files)
    for i in range(batch_size * 2):
        sorted_file_names[i] = file_path + sorted_file_names[i]
    env = DREFJSP(instance=sorted_file_names, data_source='file', batch_size=batch_size)
    return env


def validate(env, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = configs.valid_batch_size
    memory = Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, sample_parm=2, flag_train=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    a = configs.weight_makespan * env.makespan_batch / env.ini_makespan_batch
    a += configs.weight_energy * env.energy_batch / env.ini_energy_batch
    a += configs.weight_cost * env.cost_batch / env.ini_cost_batch
    a += configs.weight_demand * env.demand_batch / env.ini_demand_batch
    mul_obj = torch.mean(a)
    env.reset()
    print('validating time: ', time.time() - start, '\n')
    return mul_obj.item()
