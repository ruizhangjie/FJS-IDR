import copy
import random

from dataclasses import dataclass

import torch

from params import configs
from env.TOU_Cost import electricity_cost, cal_demand
from env.load_data import nums_extraction, load_drefjs


class DREFJSP:
    # DREFJSP 环境
    # instance表示生成算例的类或标准算例的路径序列，data_source表示算例是生成的还是标准的，batch_size表示并行运算的算例数量
    def __init__(self, instance, data_source='generate', batch_size=configs.batch_size):
        # 加载环境参数
        self.batch_size = batch_size
        self.num_jobs = configs.num_jobs
        self.num_mas = configs.num_mas

        # 加载算例
        tensors = [[] for _ in range(10)]  # 将算例读取成10个张量
        self.all_max_ops = 0  # 载入的所有算例中工序总数的最大值
        lines_time = []
        lines_proc_power = []
        lines_stan_power = []
        if data_source == 'generate':  # 生成的算例
            for i in range(self.batch_size):
                time, power_proc = instance.get_instance(i)
                lines_time.append(time)  # lines中各元素是算例中每行，算例间有换行符号
                lines_proc_power.append(power_proc)
                self.num_jobs, self.num_mas, all_max_ops = nums_extraction(lines_time[i])
                self.all_max_ops = max(self.all_max_ops, all_max_ops)
        else:  # 标准的算例
            for i in range(self.batch_size):
                # 根据路径读取文件
                with open(instance[i * 2]) as file_object:
                    line = file_object.readlines()
                    lines_proc_power.append(line)
                with open(instance[i * 2 + 1]) as file_object:
                    line = file_object.readlines()
                    lines_time.append(line)
                self.num_jobs, self.num_mas, all_max_ops = nums_extraction(lines_time[i])
                self.all_max_ops = max(self.all_max_ops, all_max_ops)

        # 以张量描述算例
        for i in range(self.batch_size):
            # 生成待机功率
            # standby_power_ma = [random.randint(configs.stanpower_min, configs.stanpower_max) for _ in
            #                     range(self.num_mas)]
            standby_power_ma = torch.tensor([10, 25, 8, 9, 21, 18, 15, 17, 14, 23])
            # standby_power_ma = torch.tensor([25, 8, 9, 17, 23])
            # standby_power_ma = torch.tensor([3, 2, 4])
            # standby_power_ma = torch.tensor([10, 25, 8, 9, 21, 18, 15, 17, 14, 23,10, 25, 8, 9, 21, 18, 15, 17, 14, 23])
            lines_stan_power.append(standby_power_ma)
            # 返回值包括：0-时间矩阵、1-功率矩阵、2-能耗矩阵、3-机器能力矩阵、4-前驱约束矩阵、5-后继约束矩阵、6-工件和工序的映射关系序列、7-各工件首道工序的idx序列、8-各工件工序数序列、9-计算每个作业沿路径的累积量矩阵
            load_data = load_drefjs(lines_time[i], lines_proc_power[i], self.num_mas, self.all_max_ops)
            for j in range(10):
                tensors[j].append(load_data[j])

        # 动态的算例描述张量
        # shape: (batch_size, num_ops, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_ops, num_mas)
        self.proc_powers_batch = torch.stack(tensors[1], dim=0)
        # shape: (batch_size, num_ops, num_mas)
        self.proc_energies_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_ops, num_mas)
        self.op_ma_adj_batch = torch.stack(tensors[3], dim=0).long()

        # 静态的算例描述张量
        # shape: (batch_size, num_mas)
        self.stan_powers_batch = torch.stack(lines_stan_power, dim=0)
        # shape: (batch_size, num_ops, num_ops)
        self.op_pre_adj_batch = torch.stack(tensors[4], dim=0)
        # shape: (batch_size, num_ops, num_ops)
        self.op_sub_adj_batch = torch.stack(tensors[5], dim=0)
        # shape: (batch_size, num_ops), represents the mapping between operations and jobs
        self.job_op_relation_batch = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.job_op_start_idx_batch = torch.stack(tensors[7], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.num_job_ops_batch = torch.stack(tensors[8], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.job_op_end_idx_batch = self.job_op_start_idx_batch + self.num_job_ops_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.sum_ops_batch = torch.sum(self.num_job_ops_batch, dim=1)
        # shape: (batch_size, num_ops, num_ops), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[9], dim=0).float()
        # 电价参数初始化
        self.low_start_time = torch.tensor(configs.low_start_time)
        self.low_end_time = torch.tensor(configs.low_end_time)
        self.high_start_time = torch.tensor(configs.high_start_time)
        self.high_end_time = torch.tensor(configs.high_end_time)
        self.low_ele_price = torch.tensor(configs.low_ele_price)
        self.mid_ele_price = torch.tensor(configs.mid_ele_price)
        self.high_ele_price = torch.tensor(configs.high_ele_price)

        # 动态变量
        # 一维序列从0到batch_size-1
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        # 深度复制可以在不更改原始数据的情况下修改新的变量，创建各工件就绪工序idx序列
        self.start_idx_step_batch = copy.deepcopy(self.job_op_start_idx_batch)
        self.cal_cumul_adj_step_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        # 机器开始加工时间和结束加工时间序列
        self.ma_start_times_batch = -99 * torch.ones(size=(self.batch_size, self.num_mas, self.all_max_ops),
                                                     dtype=torch.float32)
        self.ma_end_times_batch = torch.zeros_like(self.ma_start_times_batch, dtype=torch.float32)
        # 机器待机开始时间和结束时间序列
        self.ma_sb_start_times_batch = -99 * torch.ones(size=(self.batch_size, self.num_mas, self.all_max_ops),
                                                        dtype=torch.float32)
        self.ma_sb_end_times_batch = torch.zeros_like(self.ma_sb_start_times_batch, dtype=torch.float32)
        # 机器安排的工序idx序列
        self.ma_op_idx_batch = -99 * torch.ones(size=(self.batch_size, self.num_mas, self.all_max_ops),
                                                dtype=torch.long)
        # 机器加工功率
        self.ma_power_batch = -99 * torch.ones(size=(self.batch_size, self.num_mas, self.all_max_ops),
                                                dtype=torch.float32)
        '''
        原始特征
            工序:
                0-Status
                1-Number of neighboring machines
                2-Processing time
                3-加工功率
                4-加工能耗
                5-Start time
                6-End time
                7-加工能源成本
                8-平均加工电价
                9-Number of unscheduled operations in the job
                10-Job completion time
                11-工件完工累积加工能耗
                12-工件完工累积加工能源成本
                13-工件平均完工电价
                14-最大需量
            机器:
                0-Available time
                1-Number of neighboring operations
                2-首道工序开始加工时间
                3-待机功率
                4-累积待机次数
                5-累积待机时长
                6-累积待机能耗
                7-累积待机成本
                8-平均待机电价
            弧：
                0-加工时间
                1-加工功率
                2-加工能耗
        '''
        # Generate raw feature vectors 对应初始状态
        feat_ops_batch = torch.zeros(size=(self.batch_size, configs.in_size_op, self.all_max_ops))
        feat_mas_batch = torch.zeros(size=(self.batch_size, configs.in_size_ma, self.num_mas))
        # 调度状态默认为0
        # 可选机器数量
        feat_ops_batch[:, 1, :] = torch.count_nonzero(self.op_ma_adj_batch, dim=2)
        # 平均加工时间
        feat_ops_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_ops_batch[:, 1, :] + 1e-9)
        # 平均加工功率
        feat_ops_batch[:, 3, :] = torch.sum(self.proc_powers_batch, dim=2).div(feat_ops_batch[:, 1, :] + 1e-9)
        # 平均加工能耗
        feat_ops_batch[:, 4, :] = torch.mul(feat_ops_batch[:, 2, :], feat_ops_batch[:, 3, :])
        # bmm用于计算批矩阵乘法
        # 工序开始时间（只考虑顺序约束，0时刻所有工件都能开始）
        feat_ops_batch[:, 5, :] = torch.bmm(feat_ops_batch[:, 2, :].unsqueeze(1),
                                            self.cal_cumul_adj_batch).squeeze()
        # 工序结束时间
        feat_ops_batch[:, 6, :] = feat_ops_batch[:, 5, :] + feat_ops_batch[:, 2, :]
        # 最大需量
        self.feat_global_batch = cal_demand(feat_ops_batch[:, 3, :], feat_ops_batch[:, 5, :], feat_ops_batch[:, 6, :])
        # 工序加工能源成本
        feat_ops_batch[:, 7, :] = electricity_cost(feat_ops_batch[:, 3, :], feat_ops_batch[:, 5, :],
                                                   feat_ops_batch[:, 6, :], self.low_start_time, self.low_end_time,
                                                   self.high_start_time, self.high_end_time, self.low_ele_price,
                                                   self.mid_ele_price, self.high_ele_price)
        # 工序平均加工电价
        feat_ops_batch[:, 8, :] = feat_ops_batch[:, 7, :].div(feat_ops_batch[:, 4, :] + 1e-9)
        # 工件剩余工序数
        feat_ops_batch[:, 9, :] = convert_feat(self.num_job_ops_batch, self.job_op_relation_batch)
        # 工件结束时间，维度是(batch_size, num_jobs)
        end_time_job_batch = convert_feat(feat_ops_batch[:, 6, :], self.job_op_end_idx_batch)
        # 将工件结束时间转换成工序维度(batch_size, num_ops)
        feat_ops_batch[:, 10, :] = convert_feat(end_time_job_batch, self.job_op_relation_batch)
        # 工件完工累积加工能耗
        cumul_energy_op_batch = torch.bmm(feat_ops_batch[:, 4, :].unsqueeze(1),
                                          self.cal_cumul_adj_batch).squeeze() + feat_ops_batch[:, 4, :]
        cumul_energy_job_batch = convert_feat(cumul_energy_op_batch, self.job_op_end_idx_batch)
        feat_ops_batch[:, 11, :] = convert_feat(cumul_energy_job_batch, self.job_op_relation_batch)
        # 工件完工累积加工能源成本
        cumul_cost_op_batch = torch.bmm(feat_ops_batch[:, 7, :].unsqueeze(1),
                                        self.cal_cumul_adj_batch).squeeze() + feat_ops_batch[:, 7, :]
        cumul_cost_job_batch = convert_feat(cumul_cost_op_batch, self.job_op_end_idx_batch)
        feat_ops_batch[:, 12, :] = convert_feat(cumul_cost_job_batch, self.job_op_relation_batch)
        # 工件平均完工电价
        feat_ops_batch[:, 13, :] = feat_ops_batch[:, 12, :].div(feat_ops_batch[:, 11, :] + 1e-9)
        # 各算例最大需量
        self.demand_batch = torch.max(self.feat_global_batch, dim=1)[0]
        feat_ops_batch[:, 14, :] = self.demand_batch.unsqueeze(-1).expand(-1, self.all_max_ops)
        # 机器释放时间等默认为0
        # 可加工工序数量
        feat_mas_batch[:, 1, :] = torch.count_nonzero(self.op_ma_adj_batch, dim=1)
        feat_mas_batch[:, 2, :].fill_(-1)  # 方便更新
        feat_mas_batch[:, 3, :] = self.stan_powers_batch

        self.feat_ops_batch = feat_ops_batch
        self.feat_mas_batch = feat_mas_batch
        self.feat_edge_batch = torch.stack((self.proc_times_batch, self.proc_powers_batch, self.proc_energies_batch),
                                           dim=-1)

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        # 使用一个二进制掩码来表示哪些动作是禁止的，智能体在执行动作前需要对其进行过滤，相当于动态变量
        # shape: (batch_size, num_jobs), True for completed jobs,表示工件是否完成
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                fill_value=False)
        # all指示给定维度的所有元素是否都为True,表示算例是否完成
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        # 目标相关变量-初始值作为目标无量纲化的参数
        # 各算例最大完工时间，max函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
        self.makespan_batch = torch.max(self.feat_ops_batch[:, 10, :], dim=1)[0]  # shape: (batch_size)
        # 各算例累积能耗
        self.energy_batch = torch.sum(self.feat_ops_batch[:, 4, :], dim=1) + torch.sum(self.feat_mas_batch[:, 6, :],
                                                                                       dim=1)
        # 各算例累积能源成本
        self.cost_batch = torch.sum(self.feat_ops_batch[:, 7, :], dim=1) + torch.sum(self.feat_mas_batch[:, 7, :],
                                                                                     dim=1)

        # Save initial data for reset
        # 动态的算例描述张量
        self.ini_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.ini_proc_powers_batch = copy.deepcopy(self.proc_powers_batch)
        self.ini_proc_energies_batch = copy.deepcopy(self.proc_energies_batch)
        self.ini_op_ma_adj_batch = copy.deepcopy(self.op_ma_adj_batch)
        # 原始特征
        self.ini_feat_ops_batch = copy.deepcopy(self.feat_ops_batch)
        self.ini_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.ini_feat_global_batch = copy.deepcopy(self.feat_global_batch)
        self.ini_feat_edge_batch = copy.deepcopy(self.feat_edge_batch)
        # 目标变量
        self.ini_makespan_batch = copy.deepcopy(self.makespan_batch)
        self.ini_energy_batch = copy.deepcopy(self.energy_batch)
        self.ini_cost_batch = copy.deepcopy(self.cost_batch)
        self.ini_demand_batch = copy.deepcopy(self.demand_batch)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_ops_batch=self.feat_ops_batch, feat_mas_batch=self.feat_mas_batch,
                              feat_edge_batch=self.feat_edge_batch, op_ma_adj_batch=self.op_ma_adj_batch,
                              op_pre_adj_batch=self.op_pre_adj_batch, op_sub_adj_batch=self.op_sub_adj_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              start_idx_step_batch=self.start_idx_step_batch,
                              job_op_end_idx_batch=self.job_op_end_idx_batch,
                              sum_ops_batch=self.sum_ops_batch)
        self.ini_state = copy.deepcopy(self.state)

    def step(self, actions):
        jobs = actions[0, :]
        ops = actions[1, :]
        mas = actions[2, :]

        # 更新动态算例描述张量
        # Removed unselected O-M arcs of the scheduled operations
        remain_op_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        # 各算例选择的机器对应的O-M弧保留
        remain_op_ma_adj[self.batch_idxes, mas] = 1
        # 将各算例选择的op行整行替换
        self.op_ma_adj_batch[self.batch_idxes, ops] = remain_op_ma_adj[self.batch_idxes, :]
        # 更新加工时间矩阵
        self.proc_times_batch *= self.op_ma_adj_batch
        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines','Processing time',加工功率/能耗/能源成本
        #  维度是（batch_idxes），各算例选择的op-ma对应的时间序列
        proc_times = self.proc_times_batch[self.batch_idxes, ops, mas]
        # 更新加工功率矩阵
        self.proc_powers_batch *= self.op_ma_adj_batch
        # 更新加工能耗矩阵
        self.proc_energies_batch *= self.op_ma_adj_batch

        # 更新工件工序相关动态变量
        self.N += 1  # 各算例已调度op计数+1
        # 更新作业沿路径的累积量矩阵 for 开始时间（开始时间未调度的工序需要积累，而功率能耗等可以始终通过初始矩阵积累）
        last_ops = torch.where(ops - 1 < self.job_op_start_idx_batch[self.batch_idxes, jobs], self.all_max_ops - 1,
                               ops - 1)
        self.cal_cumul_adj_step_batch[self.batch_idxes, last_ops, :] = 0
        # 更新就绪工序
        self.start_idx_step_batch[self.batch_idxes, jobs] += 1

        # 更新工件工序原始特征
        # 更新工序级特征
        proc_powers = self.proc_powers_batch[self.batch_idxes, ops, mas]
        proc_energies = self.proc_energies_batch[self.batch_idxes, ops, mas]
        # 计算各算例选择工序的开始时间
        job_rdy_time = self.feat_ops_batch[self.batch_idxes, 5, ops]
        ma_rdy_time = self.feat_mas_batch[self.batch_idxes, 0, mas]
        start_time_action = torch.max(ma_rdy_time, job_rdy_time)
        end_time_action = start_time_action + proc_times
        # 计算cost时为了不单独算一次把当前op也纳入未调度
        is_scheduled_cost = self.feat_ops_batch[self.batch_idxes, 0, :]
        un_scheduled_cost = 1 - is_scheduled_cost
        self.feat_ops_batch[self.batch_idxes, :7, ops] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             proc_times, proc_powers, proc_energies, start_time_action,
             end_time_action), dim=1)
        # 迭代更新选择工件未调度的工序的特征
        is_scheduled = self.feat_ops_batch[self.batch_idxes, 0, :]
        # 已调度工序是实际时间未调度还是平均时间
        mean_proc_time = self.feat_ops_batch[self.batch_idxes, 2, :]
        start_times = self.feat_ops_batch[self.batch_idxes, 5, :] * is_scheduled  # real start time of scheduled opes
        un_scheduled = 1 - is_scheduled  # unscheduled opes
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                                   self.cal_cumul_adj_step_batch[self.batch_idxes, :, :]).squeeze() \
                         * un_scheduled  # estimate start time of unscheduled opes
        # 根据已调度工序更新所有节点开始时间
        self.feat_ops_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        self.feat_ops_batch[self.batch_idxes, 6, :] = self.feat_ops_batch[self.batch_idxes, 5, :] + self.feat_ops_batch[
                                                                                                    self.batch_idxes, 2,
                                                                                                    :]
        # 以加工功率更新需量
        self.feat_global_batch = cal_demand(self.feat_ops_batch[:, 3, :], self.feat_ops_batch[:, 5, :],
                                            self.feat_ops_batch[:, 6, :])
        # 根据已调度工序更新所有节点加工能源成本，计算1次
        powers = self.feat_ops_batch[self.batch_idxes, 3, :] * un_scheduled_cost  # 实际功率
        real_cost = self.feat_ops_batch[self.batch_idxes, 7, :] * is_scheduled_cost  # 实际成本
        start_times_cost = self.feat_ops_batch[self.batch_idxes, 5, :] * un_scheduled_cost
        end_times_cost = self.feat_ops_batch[self.batch_idxes, 6, :] * un_scheduled_cost
        estimate_costs = electricity_cost(powers, start_times_cost, end_times_cost, self.low_start_time,
                                          self.low_end_time,
                                          self.high_start_time, self.high_end_time, self.low_ele_price,
                                          self.mid_ele_price, self.high_ele_price)
        self.feat_ops_batch[self.batch_idxes, 7, :] = real_cost + estimate_costs
        self.feat_ops_batch[self.batch_idxes, 8, :] = self.feat_ops_batch[self.batch_idxes, 7, :].div(
            self.feat_ops_batch[self.batch_idxes, 4, :] + 1e-9)
        # 更新工件级特征
        start_op = self.job_op_start_idx_batch[self.batch_idxes, jobs]
        end_op = self.job_op_end_idx_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_ops_batch[self.batch_idxes[i], 9, start_op[i]:end_op[i] + 1] -= 1

        end_time_job_batch = convert_feat(self.feat_ops_batch[self.batch_idxes, 6, :], self.job_op_end_idx_batch[
                                                                                       self.batch_idxes, :])
        # 根据已调度工序更新所有工件完工时间
        self.feat_ops_batch[self.batch_idxes, 10, :] = convert_feat(end_time_job_batch, self.job_op_relation_batch[
                                                                                        self.batch_idxes, :])
        # 工件完工累积加工能耗
        cumul_energy_op_batch = torch.bmm(self.feat_ops_batch[self.batch_idxes, 4, :].unsqueeze(1),
                                          self.cal_cumul_adj_batch[self.batch_idxes]).squeeze() + self.feat_ops_batch[
                                                                                                  self.batch_idxes, 4,
                                                                                                  :]
        cumul_energy_job_batch = convert_feat(cumul_energy_op_batch, self.job_op_end_idx_batch[self.batch_idxes])
        self.feat_ops_batch[self.batch_idxes, 11, :] = convert_feat(cumul_energy_job_batch,
                                                                    self.job_op_relation_batch[self.batch_idxes])
        # 工件完工累积加工能源成本
        cumul_cost_op_batch = torch.bmm(self.feat_ops_batch[self.batch_idxes, 7, :].unsqueeze(1),
                                        self.cal_cumul_adj_batch[self.batch_idxes]).squeeze() + self.feat_ops_batch[
                                                                                                self.batch_idxes, 7, :]
        cumul_cost_job_batch = convert_feat(cumul_cost_op_batch, self.job_op_end_idx_batch[self.batch_idxes])
        self.feat_ops_batch[self.batch_idxes, 12, :] = convert_feat(cumul_cost_job_batch,
                                                                    self.job_op_relation_batch[self.batch_idxes])
        self.feat_ops_batch[self.batch_idxes, 13, :] = self.feat_ops_batch[self.batch_idxes, 12, :].div(
            self.feat_ops_batch[self.batch_idxes, 11, :] + 1e-9)

        # 更新机器相关动态变量
        mask = self.ma_start_times_batch[self.batch_idxes, mas, :] < 0
        indices = torch.argmax(mask.int(), dim=1)
        self.ma_start_times_batch[self.batch_idxes, mas, indices] = start_time_action
        self.ma_end_times_batch[self.batch_idxes, mas, indices] = end_time_action
        self.ma_op_idx_batch[self.batch_idxes, mas, indices] = ops
        self.ma_power_batch[self.batch_idxes, mas, indices] = proc_powers
        ava_time_pre = self.feat_mas_batch[self.batch_idxes, 0, mas]
        mask1 = 1 - torch.eq(ava_time_pre, 0).int()
        differ = start_time_action - ava_time_pre
        standby_slot_action = differ * mask1  # 先计算待机时间再更新
        mask2 = self.ma_sb_start_times_batch[self.batch_idxes, mas, :] < 0
        indices1 = torch.argmax(mask2.int(), dim=1)
        sb_start_time_action = torch.where(standby_slot_action != 0, ava_time_pre, torch.tensor(-99.))
        sb_end_time_action = torch.where(standby_slot_action != 0, start_time_action, torch.tensor(0.))
        self.ma_sb_start_times_batch[self.batch_idxes, mas, indices1] = sb_start_time_action
        self.ma_sb_end_times_batch[self.batch_idxes, mas, indices1] = sb_end_time_action

        # 以待机功率更新需量
        start_flat = self.ma_sb_start_times_batch.flatten(1)
        end_flat = self.ma_sb_end_times_batch.flatten(1)
        power_flat = self.feat_mas_batch[:, 3, :].unsqueeze(-1).expand(-1, -1, self.all_max_ops).flatten(1)
        mask3 = start_flat > 0
        power_flat = torch.where(mask3, power_flat, torch.tensor(0.))
        sb_demand = cal_demand(power_flat, start_flat, end_flat)
        self.feat_global_batch = self.feat_global_batch + sb_demand
        # 完整的需量更新工序特征
        demand = torch.max(self.feat_global_batch[self.batch_idxes], dim=1)[0]
        demand_copy = demand.unsqueeze(-1).expand(-1, self.all_max_ops)
        demand_copy = demand_copy * un_scheduled
        old_demand = self.feat_ops_batch[self.batch_idxes, 14, :] * is_scheduled
        self.feat_ops_batch[self.batch_idxes, 14, :] = old_demand + demand_copy

        # 更新机器初始特征
        self.feat_mas_batch[self.batch_idxes, 0, mas] = end_time_action
        self.feat_mas_batch[self.batch_idxes, 1, :] = torch.count_nonzero(self.op_ma_adj_batch[self.batch_idxes, :, :],
                                                                          dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 2, mas] = torch.where(self.feat_mas_batch[self.batch_idxes, 2, mas] < 0,
                                                                    start_time_action,
                                                                    self.feat_mas_batch[self.batch_idxes, 2, mas])
        standby_count_action = torch.where(standby_slot_action != 0, torch.tensor(1), torch.tensor(0))
        standby_energies_action = torch.mul(self.stan_powers_batch[self.batch_idxes, mas], standby_slot_action)
        standby_cost_action = electricity_cost(self.stan_powers_batch[self.batch_idxes, mas], ava_time_pre,
                                               start_time_action, self.low_start_time, self.low_end_time,
                                               self.high_start_time, self.high_end_time, self.low_ele_price,
                                               self.mid_ele_price, self.high_ele_price)
        standby_cost_action *= mask1
        self.feat_mas_batch[self.batch_idxes, 4, mas] = self.feat_mas_batch[
                                                            self.batch_idxes, 4, mas] + standby_count_action
        self.feat_mas_batch[self.batch_idxes, 5, mas] = self.feat_mas_batch[
                                                            self.batch_idxes, 5, mas] + standby_slot_action
        self.feat_mas_batch[self.batch_idxes, 6, mas] = self.feat_mas_batch[self.batch_idxes, 6, mas] + \
                                                        standby_energies_action
        self.feat_mas_batch[self.batch_idxes, 7, mas] = self.feat_mas_batch[self.batch_idxes, 7, mas] + \
                                                        standby_cost_action
        self.feat_mas_batch[self.batch_idxes, 8, mas] = self.feat_mas_batch[self.batch_idxes, 7, mas].div(
            self.feat_mas_batch[self.batch_idxes, 6, mas] + 1e-9)

        # 更新弧初始特征
        self.feat_edge_batch = torch.stack((self.proc_times_batch, self.proc_powers_batch, self.proc_energies_batch),
                                           dim=-1)

        # 更新mask
        self.mask_job_finish_batch = torch.where(self.start_idx_step_batch == self.job_op_end_idx_batch + 1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)

        # 计算奖励
        max_makespan_batch = torch.max(self.feat_ops_batch[:, 10, :], dim=1)[0]
        reward_makespan_batch = (self.makespan_batch - max_makespan_batch) / self.ini_makespan_batch
        max_energy_batch = torch.sum(self.feat_ops_batch[:, 4, :], dim=1) + torch.sum(self.feat_mas_batch[:, 6, :],
                                                                                      dim=1)
        reward_energy_batch = (self.energy_batch - max_energy_batch) / self.ini_energy_batch
        max_cost_batch = torch.sum(self.feat_ops_batch[:, 7, :], dim=1) + torch.sum(self.feat_mas_batch[:, 7, :],
                                                                                    dim=1)
        reward_cost_batch = (self.cost_batch - max_cost_batch) / self.ini_cost_batch
        max_demand_batch = torch.max(self.feat_global_batch, dim=1)[0]
        reward_demand_batch = (self.demand_batch - max_demand_batch) / self.ini_demand_batch
        self.reward_batch = configs.weight_makespan * reward_makespan_batch + configs.weight_energy * reward_energy_batch + configs.weight_cost * reward_cost_batch + configs.weight_demand * reward_demand_batch
        self.makespan_batch = max_makespan_batch
        self.energy_batch = max_energy_batch
        self.cost_batch = max_cost_batch
        self.demand_batch = max_demand_batch

        # Update the vector for uncompleted instances
        mask_finish = (self.N + 1) <= self.sum_ops_batch
        if ~(mask_finish.all()):
            # 有任意实例调度完成后更新batch_idxes，完成的实例被移除
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        # Update state of the environment
        self.state.update(self.batch_idxes, self.feat_ops_batch, self.feat_mas_batch, self.feat_edge_batch,
                          self.op_ma_adj_batch, self.mask_job_finish_batch,
                          self.start_idx_step_batch)

        return self.state, self.reward_batch, self.done_batch

    def reset(self):
        '''
        Reset the environment to its initial state，静态张量无视
        '''
        # 动态的算例描述张量
        self.proc_times_batch = copy.deepcopy(self.ini_proc_times_batch)
        self.proc_powers_batch = copy.deepcopy(self.ini_proc_powers_batch)
        self.proc_energies_batch = copy.deepcopy(self.ini_proc_energies_batch)
        self.op_ma_adj_batch = copy.deepcopy(self.ini_op_ma_adj_batch)
        # 原始特征
        self.feat_ops_batch = copy.deepcopy(self.ini_feat_ops_batch)
        self.feat_mas_batch = copy.deepcopy(self.ini_feat_mas_batch)
        self.feat_global_batch = copy.deepcopy(self.ini_feat_global_batch)
        self.feat_edge_batch = copy.deepcopy(self.ini_feat_edge_batch)
        # 动态变量重新初始化
        self.batch_idxes = torch.arange(self.batch_size)
        self.N = torch.zeros(self.batch_size).int()
        self.start_idx_step_batch = copy.deepcopy(self.job_op_start_idx_batch)
        self.cal_cumul_adj_step_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.ma_start_times_batch = -99 * torch.ones(size=(self.batch_size, self.num_mas, self.all_max_ops),
                                                     dtype=torch.float32)
        self.ma_end_times_batch = torch.zeros_like(self.ma_start_times_batch, dtype=torch.float32)
        self.ma_sb_start_times_batch = -99 * torch.ones(size=(self.batch_size, self.num_mas, self.all_max_ops),
                                                        dtype=torch.float32)
        self.ma_sb_end_times_batch = torch.zeros_like(self.ma_sb_start_times_batch, dtype=torch.float32)
        self.ma_op_idx_batch = -99 * torch.ones(size=(self.batch_size, self.num_mas, self.all_max_ops),
                                                dtype=torch.long)
        self.ma_power_batch = -99 * torch.ones(size=(self.batch_size, self.num_mas, self.all_max_ops),
                                                dtype=torch.float32)
        self.state = copy.deepcopy(self.ini_state)
        # 掩码重新初始化
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                fill_value=False)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        # 目标相关变量初始化
        self.makespan_batch = copy.deepcopy(self.ini_makespan_batch)
        self.energy_batch = copy.deepcopy(self.ini_energy_batch)
        self.cost_batch = copy.deepcopy(self.ini_cost_batch)
        self.demand_batch = copy.deepcopy(self.ini_demand_batch)
        return self.state


def convert_feat(feat_in, feat_out):
    # 将工件特征和工序特征进行相互转换，通过替换第二维度
    return feat_in.gather(1, feat_out)


@dataclass
class EnvState:
    # static
    batch_idxes: torch.Tensor = None
    op_pre_adj_batch: torch.Tensor = None
    op_sub_adj_batch: torch.Tensor = None
    job_op_end_idx_batch: torch.Tensor = None
    sum_ops_batch: torch.Tensor = None

    # dynamic
    feat_ops_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    feat_edge_batch: torch.Tensor = None
    op_ma_adj_batch: torch.Tensor = None

    mask_job_finish_batch: torch.Tensor = None
    start_idx_step_batch: torch.Tensor = None

    def update(self, batch_idxes, feat_ops_batch, feat_mas_batch, feat_edge_batch, op_ma_adj_batch,
               mask_job_finish_batch, start_idx_step_batch):
        self.batch_idxes = batch_idxes
        self.feat_ops_batch = feat_ops_batch
        self.feat_mas_batch = feat_mas_batch
        self.feat_edge_batch = feat_edge_batch
        self.op_ma_adj_batch = op_ma_adj_batch

        self.mask_job_finish_batch = mask_job_finish_batch
        self.start_idx_step_batch = start_idx_step_batch
