import random

import torch

from env.TOU_Cost import electricity_cost
from params import configs


def PDR1(state):
    job, op = FIFO(state)
    ma = SPT(op, state)
    return torch.stack((job, op, ma), dim=0)


def PDR2(state, ma_stand_power):
    job, op = FIFO(state)
    ma = LCP(op, state, ma_stand_power)
    return torch.stack((job, op, ma), dim=0)


def PDR3(state, ma_stand_power):
    job, op = FIFO(state)
    ma = LCT(op, state, ma_stand_power)
    return torch.stack((job, op, ma), dim=0)


def PDR4(state):
    job, op = FIFO(state)
    ma = SPT(op, state)
    return torch.stack((job, op, ma), dim=0)


def Random(state, ma_stand_power):
    probabilities = torch.tensor(
        [configs.weight_makespan, configs.weight_energy, configs.weight_cost, configs.weight_demand])
    sampled_number = torch.multinomial(probabilities, 1, replacement=True)
    function_mapping = {
        0: PDR1(state),
        1: PDR2(state, ma_stand_power),
        2: PDR3(state, ma_stand_power),
        3: PDR4(state)
    }
    return function_mapping[sampled_number.item()]


def get_value(value):
    min_values, _ = value.min(dim=1, keepdim=True)
    mask = value == min_values
    min_positions = torch.where(mask)
    min_indices = []
    for i in range(value.shape[0]):
        indices = min_positions[1][min_positions[0] == i]
        random_index = indices[torch.randint(len(indices), (1,))].item()
        min_indices.append(random_index)
    return torch.tensor(min_indices)


def FIFO(state):
    value = state.feat_ops_batch[state.batch_idxes, 5]
    start_idx = state.start_idx_step_batch
    job_mask = state.mask_job_finish_batch
    start_idx[job_mask] = 0
    value = value.gather(1, start_idx)
    value[job_mask] = float('inf')
    job = get_value(value)
    op = start_idx[state.batch_idxes, job]
    return job, op


def SPT(op, state):
    value = state.feat_edge_batch[state.batch_idxes, op, :, 0]
    ma_mask = state.op_ma_adj_batch[state.batch_idxes, op] == 0
    value[ma_mask] = float('inf')
    ma = get_value(value)
    return ma


def SPP(op, state):
    value = state.feat_edge_batch[state.batch_idxes, op, :, 1]
    ma_mask = state.op_ma_adj_batch[state.batch_idxes, op] == 0
    value[ma_mask] = float('inf')
    ma = get_value(value)
    return ma


def LCP(op, state, ma_stand_power):
    job_time = state.feat_ops_batch[state.batch_idxes, 5, op]
    ma_time = state.feat_mas_batch[state.batch_idxes, 0, :]
    start_time = torch.where(ma_time > job_time, ma_time, job_time)
    pro_energy = state.feat_edge_batch[state.batch_idxes, op, :, 2]
    std_energy = ma_stand_power * (start_time - ma_time)
    total_energy = pro_energy + std_energy
    ma_mask = state.op_ma_adj_batch[state.batch_idxes, op] == 0
    total_energy[ma_mask] = float('inf')
    ma = get_value(total_energy)
    return ma


def LCT(op, state, ma_stand_power):
    job_time = state.feat_ops_batch[state.batch_idxes, 5, op]
    ma_time = state.feat_mas_batch[state.batch_idxes, 0, :]
    start_time = torch.where(ma_time > job_time, ma_time, job_time)
    end_time = start_time + state.feat_edge_batch[state.batch_idxes, op, :, 0]
    power = state.feat_edge_batch[state.batch_idxes, op, :, 1]
    pro_cost = electricity_cost(power, start_time, end_time, torch.tensor(configs.low_start_time),
                                torch.tensor(configs.low_end_time),
                                torch.tensor(configs.high_start_time), torch.tensor(configs.high_end_time),
                                torch.tensor(configs.low_ele_price),
                                torch.tensor(configs.mid_ele_price), torch.tensor(configs.high_ele_price))
    std_cost = electricity_cost(ma_stand_power, ma_time, start_time, torch.tensor(configs.low_start_time),
                                torch.tensor(configs.low_end_time),
                                torch.tensor(configs.high_start_time), torch.tensor(configs.high_end_time),
                                torch.tensor(configs.low_ele_price),
                                torch.tensor(configs.mid_ele_price), torch.tensor(configs.high_ele_price))
    total_cost = pro_cost + std_cost
    ma_mask = state.op_ma_adj_batch[state.batch_idxes, op] == 0
    total_cost[ma_mask] = float('inf')
    ma = get_value(total_cost)
    return ma
