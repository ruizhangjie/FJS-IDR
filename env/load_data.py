import numpy as np
import torch


def nums_extraction(lines):
    # 从算例中读取工件数、机器数和总工序数
    num_ops = 0
    # 生成算例中最后的元素是换行符
    for i in range(1, len(lines)):
        num_ops += int(lines[i].strip().split()[0]) if lines[i] != "\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    return num_jobs, num_mas, num_ops

def load_drefjs(lines_time, lines_power, num_mas, num_ops):
    # 读取算例输出初始特征张量
    flag = 0
    # 加工时间矩阵,shape:(num_ops,num_mas)
    matrix_proc_time = torch.zeros(size=(num_ops, num_mas), dtype=torch.float32)
    # 加工功率矩阵,shape:(num_ops,num_mas)
    matrix_proc_power = torch.zeros(size=(num_ops, num_mas), dtype=torch.float32)
    # 工序顺序关系矩阵,shape:(num_ops,num_ops)
    matrix_pre_proc = torch.full(size=(num_ops, num_ops), dtype=torch.bool, fill_value=False)
    # 计算每个作业沿路径的累积量矩阵,shape:(num_ops,num_ops)
    matrix_cal_cumul = torch.zeros(size=(num_ops, num_ops)).int()
    num_job_ops = []  # A list of the number of operations for each job
    job_op_relation = np.array([])
    job_op_start_idx = []  # The id of the first operation of each job
    # Parse data line by line
    for line in lines_time:
        # first line 首行不处理
        if flag == 0:
            flag += 1
        # last line 尾行结束
        elif line == "\n":
            break
        # other
        else:
            op_start_idx = int(sum(num_job_ops))  # The id of the first operation of this job
            job_op_start_idx.append(op_start_idx)
            # Detect information of this job and return the number of operations
            num_op = fill_matrix_job(line, op_start_idx, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
            num_job_ops.append(num_op)
            job_op_relation = np.concatenate((job_op_relation, np.ones(num_op) * (flag - 1)))
            flag += 1
    flag = 0
    num_job_ops = []
    # 填充功率矩阵
    for line in lines_power:
        # first line 首行不处理
        if flag == 0:
            flag += 1
        # last line 尾行结束
        elif line == "\n":
            break
        else:
            op_start_idx = int(sum(num_job_ops))  # The id of the first operation of this job
            num_op = fill_power_matrix(line, op_start_idx, matrix_proc_power)
            num_job_ops.append(num_op)
    #  机器能力矩阵，shape:(num_ops,num_mas)
    matrix_op_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    #  加工能耗矩阵，shape:(num_ops,num_mas)
    matrix_proc_energy = torch.mul(matrix_proc_time, matrix_proc_power)
    # Fill zero if the operations are insufficient (for parallel computation)，shape:(num_ops)，各元素的值为工序对应的工件idx
    job_op_relation = np.concatenate((job_op_relation, np.zeros(num_ops - job_op_relation.size)))
    return matrix_proc_time, matrix_proc_power, matrix_proc_energy, matrix_op_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
        torch.tensor(job_op_relation).int(), torch.tensor(job_op_start_idx).int(), \
        torch.tensor(num_job_ops).int(), matrix_cal_cumul

def fill_matrix_job(line, op_start_idx, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
    # 以工件为单位依次填充相关特征矩阵
    # 按照空格分隔
    line_time_split = line.split()
    flag = 0  # 指针
    flag_ma = 1  # 1-处理机器，0-处理时间/功率
    flag_new_op = 1  # 新工序指针
    idx_op = -1  # 正在处理的工序idx
    num_ops = 0  # Store the number of operations of this job
    num_op_mas = np.array([])  # Store the number of processable machines for each operation of this job
    ma = 0
    # i代表具体数字
    for i in line_time_split:
        x = float(i)
        # The first number indicates the number of operations of this job
        if flag == 0:
            num_ops = x
            flag += 1
        # new operation detected
        elif flag == flag_new_op:
            idx_op += 1
            flag_new_op += x * 2 + 1
            num_op_mas = np.append(num_op_mas, x)
            # 非最后的工序才有前后关系
            if idx_op != num_ops-1:
                matrix_pre_proc[idx_op+op_start_idx][idx_op+op_start_idx+1] = True
            # 非首道工序
            if idx_op != 0:
                # size(0)返回张量的第一个维度大小
                vector = torch.zeros(matrix_cal_cumul.size(0))
                vector[idx_op+op_start_idx-1] = 1
                matrix_cal_cumul[:, idx_op+op_start_idx] = matrix_cal_cumul[:, idx_op+op_start_idx-1]+vector
            flag += 1
        # not proc_time (machine)
        elif flag_ma == 1:
            ma = int(x-1)
            flag += 1
            flag_ma = 0
        # proc_time
        else:
            matrix_proc_time[idx_op+op_start_idx][ma] = x
            flag += 1
            flag_ma = 1
    return int(num_ops)

def fill_power_matrix(line_power, op_start_idx, matrix_proc_power):
    line_power_split = line_power.split()
    flag = 0  # 指针
    flag_ma = 1  # 1-处理机器，0-处理时间/功率
    flag_new_op = 1  # 新工序指针
    idx_op = -1  # 正在处理的工序idx
    ma = 0
    num_ops = 0  # Store the number of operations of this job
    for i in line_power_split:
        x = float(i)
        if flag == 0:
            num_ops = x
            flag += 1
        elif flag == flag_new_op:
            idx_op += 1
            flag_new_op += x * 2 + 1
            flag += 1
        elif flag_ma == 1:
            ma = int(x - 1)
            flag += 1
            flag_ma = 0
        # proc_power
        else:
            matrix_proc_power[idx_op + op_start_idx][ma] = x
            flag += 1
            flag_ma = 1
    return int(num_ops)
