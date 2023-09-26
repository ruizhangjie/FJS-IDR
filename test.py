import copy
import os
import time

import numpy as np
import pandas as pd
import torch

from PPO_Model.PPO import Memory, PPO
from dispatch_rule import PDR1, PDR4, PDR2, PDR3, Random
from env.DREFJSP_Env import DREFJSP
from params import configs
from plot_gantt import initialize_plt, plt_demand


def main():
    # PyTorch initialization
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None,
                           sci_mode=False)

    # Load config and init objects
    batch_size = 1
    data_path = "./data_test/{0}/".format(configs.data_path)
    test_files = os.listdir(data_path)
    sorted_file_names = sorted(test_files)
    for i in range(len(sorted_file_names)):
        sorted_file_names[i] = data_path + sorted_file_names[i]
    mod_files = os.listdir('./models/')[:]
    priority_dispatch_rule = ['SPT', 'SPP','Random']
    # priority_dispatch_rule = ['SPT', 'SPP', 'LCP', 'LCT','Random']
    # for item in priority_dispatch_rule:
    #     mod_files.append(item)
    memories = Memory()
    model = PPO(batch_size)

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './record/test_{0}'.format(str_time)
    os.makedirs(save_path)
    writer = pd.ExcelWriter(
        '{0}/objs_{1}.xlsx'.format(save_path, str_time))  # mul_obj data storage path
    writer_time = pd.ExcelWriter('{0}/time_{1}.xlsx'.format(save_path, str_time))  # time data storage path
    file_name = ['{0}_{1}'.format(configs.data_path, i + 1) for i in range(configs.num_repeat)]
    data_file = pd.DataFrame(file_name, columns=["file_count"])
    data_file.to_excel(writer, sheet_name='Sheet1', index=False)
    data_file.to_excel(writer_time, sheet_name='Sheet1', index=False)

    # Rule-by-rule (model-by-model) testing
    start = time.time()
    for i_model in range(len(mod_files)):
        rule = mod_files[i_model]
        # Load trained model
        if rule.endswith('.pt'):
            if device.type == 'cuda':
                model_CKPT = torch.load('./models/' + mod_files[i_model])
            else:
                model_CKPT = torch.load('./models/' + mod_files[i_model], map_location='cpu')
            print('\nloading checkpoint:', rule)
            model.policy.load_state_dict(model_CKPT)
            model.policy_old.load_state_dict(model_CKPT)
        print('model:', rule)

        # Schedule instance by instance
        step_time_last = time.time()
        mul_objs = []
        times = []
        env = DREFJSP(instance=sorted_file_names, data_source='file', batch_size=1)
        for num in range(configs.num_repeat):
            if rule.endswith('.pt'):
                mul_obj_batch, spend_time = schedule(env, model, memories, save_path,
                                                     num + 1)
            else:
                mul_obj_batch, spend_time = schedule_dispatch_rule(env, rule)
            mul_objs.append(mul_obj_batch)
            times.append(spend_time)
            print("finish env {0};".format(num + 1), "spend_time: ", spend_time)
            env.reset()

        print("model_spend_time: ", time.time() - step_time_last)
        # Save mul_obj and time data to files
        mul_objs_array = np.array(mul_objs)
        col_name = [rule + '_Makespan', rule + '_Cost', rule + '_Energy', rule + '_Demand', rule + '_Fitness']
        # col_name = [rule + '_Fitness']
        data = pd.DataFrame(mul_objs_array, columns=col_name)
        data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=i_model * 5 + 1)
        # data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=i_model + 1)
        times_array = np.array(times)
        data = pd.DataFrame(times_array, columns=[rule])
        data.to_excel(writer_time, sheet_name='Sheet1', index=False, startcol=i_model + 1)
    writer.close()
    writer_time.close()
    print("total_spend_time: ", time.time() - start)


def schedule(env, model, memories, path, num):
    # Get state and completion signal
    state = copy.deepcopy(env.state)
    done = False  # Unfinished at the beginning
    last_time = time.time()
    while ~done:
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, flag_train=False)
        state1, _, dones = env.step(actions)  # environment transit
        state = copy.deepcopy(state1)
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    a = configs.weight_makespan * env.makespan_batch / env.ini_makespan_batch
    a += configs.weight_energy * env.energy_batch / env.ini_energy_batch
    a += configs.weight_cost * env.cost_batch / env.ini_cost_batch
    a += configs.weight_demand * env.demand_batch / env.ini_demand_batch
    mul_obj_batch = [env.makespan_batch.item(), env.cost_batch.item(), env.energy_batch.item(), env.demand_batch.item(),
                     a.item()]
    # mul_obj_batch = [a.item()]
    # 规模过大后不推荐绘制完整gantt图
    # ma_start = env.ma_start_times_batch.squeeze()
    # ma_end = env.ma_end_times_batch.squeeze()
    # ma_op = env.ma_op_idx_batch.squeeze()
    # ma_power = env.ma_power_batch.squeeze()
    # ma_sb_start = env.ma_sb_start_times_batch.squeeze()
    # ma_sb_end = env.ma_sb_end_times_batch.squeeze()
    # ma_sb_power = env.stan_powers_batch.squeeze()
    # max_time = initialize_plt(env.num_mas, env.num_jobs, ma_op, ma_start,
    #                           ma_end, env.job_op_start_idx_batch.squeeze(), path, num)
    # plt_demand(env.num_mas, env.num_jobs, ma_start, ma_end, ma_power, ma_sb_start, ma_sb_end, ma_sb_power, path, num,
    #            max_time)

    return mul_obj_batch, spend_time


def schedule_dispatch_rule(env, rule):
    done = False  # Unfinished at the beginning
    last_time = time.time()
    state = copy.deepcopy(env.state)  # 避免state被修改
    while ~done:
        if rule == 'SPT':
            actions = PDR1(state)
        elif rule == 'SPP':
            actions = PDR4(state)
        elif rule == 'LCP':
            actions = PDR2(state, env.stan_powers_batch)
        elif rule == 'LCT':
            actions = PDR3(state, env.stan_powers_batch)
        else:
            actions = Random(state,env.stan_powers_batch)
        state1, _, dones = env.step(actions)  # environment transit
        state = copy.deepcopy(state1)
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    a = configs.weight_makespan * env.makespan_batch / env.ini_makespan_batch
    a += configs.weight_energy * env.energy_batch / env.ini_energy_batch
    a += configs.weight_cost * env.cost_batch / env.ini_cost_batch
    a += configs.weight_demand * env.demand_batch / env.ini_demand_batch
    mul_obj_batch = [env.makespan_batch.item(), env.cost_batch.item(), env.energy_batch.item(), env.demand_batch.item(),
                     a.item()]
    # mul_obj_batch = [a.item()]
    return mul_obj_batch, spend_time

if __name__ == '__main__':
    main()
