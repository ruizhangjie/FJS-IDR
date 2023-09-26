import random

import torch

from params import configs


class InstanceGenerator:

    def __init__(self, num_jobs, num_mas, num_job_ops=None, path='../data_dev/',
                 flag_same_op=False, flag_doc=False):
        if num_job_ops is None:
            num_job_ops = []
        self.flag_doc = flag_doc  # Whether save the instance to a file
        self.flag_same_op = flag_same_op  # 算例中所有工件的工序数是否相同
        self.num_job_ops = num_job_ops  # 算例中所有工件的工序数序列
        self.path = path  # Instance save path (relative path)
        self.num_jobs = num_jobs  # 工件数
        self.num_mas = num_mas  # 机器数

        self.mas_per_op_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_op_max = num_mas
        self.ops_per_job_min = int(
            num_mas * (1 - configs.ops_per_job_dev))  # The minimum number of operations for a job
        self.ops_per_job_max = int(num_mas * (1 + configs.ops_per_job_dev))
        self.proctime_per_op_min = configs.proctime_per_op_min  # Minimum average processing time
        self.proctime_per_op_max = configs.proctime_per_op_max
        self.procpower_per_op_min = configs.procpower_per_op_min  # Minimum average processing power
        self.procpower_per_op_max = configs.procpower_per_op_max
        self.proctime_dev = configs.proctime_dev  # 各工序加工时间采样的松弛系数
        self.procpower_dev = configs.procpower_dev  # 各工序加工功率采样的松弛系数
        self.stanpower_min = configs.stanpower_min  # Minimum standby power
        self.stanpower_max = configs.stanpower_max

    def get_instance(self, idx=0):
        # idx表示算例的索引
        if not self.flag_same_op:
            # 生成各工件的工序数序列
            self.num_job_ops = [random.randint(self.ops_per_job_min, self.ops_per_job_max) for _ in
                                range(self.num_jobs)]
        self.sum_ops = sum(self.num_job_ops)  # 算例中所有工件的工序总数
        # 各工序可选机器数序列
        self.num_op_mas = [random.randint(self.mas_per_op_min, self.mas_per_op_max) for _ in range(self.sum_ops)]
        self.sum_mas = sum(self.num_op_mas)  # 所有工序可选机器总数 
        # 各工序可选机器idx序列
        self.op_ma_idx = []
        for val in self.num_op_mas:
            # 根据各工序可选机器数量进行机器抽样，输出机器idx序列
            self.op_ma_idx = self.op_ma_idx + sorted(random.sample(range(1, self.num_mas + 1), val))
        # 各工序在可选机器上加工时间序列
        self.proc_time = []
        # 各工序在可选机器上加工功率序列
        self.proc_power = []
        # 各工序加工时间平均值序列
        self.proc_time_mean = [round(random.uniform(self.proctime_per_op_min, self.proctime_per_op_max), 1) for _ in
                               range(self.sum_ops)]
        for i in range(self.sum_ops):
            low_bound_time = max(self.proctime_per_op_min, round(self.proc_time_mean[i] * (1 - self.proctime_dev), 1))
            high_bound_time = min(self.proctime_per_op_max, round(self.proc_time_mean[i] * (1 + self.proctime_dev), 1))
            # 生成各工序在对应可选机器上的时间序列
            proc_time_exact = [round(random.uniform(low_bound_time, high_bound_time), 1) for _ in
                               range(self.num_op_mas[i])]
            self.proc_time = self.proc_time + proc_time_exact
        # 各工序加工功率平均值序列
        self.proc_power_mean = [round(random.uniform(self.procpower_per_op_min, self.procpower_per_op_max), 1) for _ in
                                range(self.sum_ops)]
        for i in range(self.sum_ops):
            low_bound_power = max(self.procpower_per_op_min,
                                  round(self.proc_power_mean[i] * (1 - self.procpower_dev), 1))
            high_bound_power = min(self.procpower_per_op_max,
                                   round(self.proc_power_mean[i] * (1 + self.proctime_dev), 1))
            # 生成各工序在对应可选机器上的功率序列
            proc_power_exact = [round(random.uniform(low_bound_power, high_bound_power), 1) for _ in
                                range(self.num_op_mas[i])]
            self.proc_power = self.proc_power + proc_power_exact
        # 各工件首道工序的idx
        self.job_op_start_idx = [sum(self.num_job_ops[0:i]) for i in range(self.num_jobs)]
        # 各工序首个机器的idx
        self.op_ma_start_idx = [sum(self.num_op_mas[0:i]) for i in range(self.sum_ops)]
        # 算例标题行---工件 机器 平均可选机器数
        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_mas, self.sum_mas / self.sum_ops)
        lines_time = []  # 时间部分作为方法输出
        lines_power = []  # 功率部分作为方法输出
        lines_time_doc = []  # 时间部分作为文件生成
        lines_power_doc = []  # 功率部分作为文件生成
        lines_time.append(line0)
        lines_power.append(line0)
        lines_time_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_mas, self.sum_mas / self.sum_ops))
        lines_power_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_mas, self.sum_mas / self.sum_ops))
        # 循环工件
        for i in range(self.num_jobs):
            flag = 0  # 指针
            flag_ma = 1  # 1-写机器，0-写时间/功率
            flag_new_op = 1  # 新工序指针 
            idx_op = -1  # 正在处理的工序idx 
            idx_ma = 0  # 正在处理的机器idx 
            line_time = []
            line_power = []
            # 工件可选机器总和
            sum_ma = sum(self.num_op_mas[self.job_op_start_idx[i]:(self.job_op_start_idx[i] + self.num_job_ops[i])])
            sum_handled_ma = 0  # 已写入的机器数量
            while True:
                if flag == 0:
                    # 第一位数字：工序总数
                    line_time.append(self.num_job_ops[i])
                    line_power.append(self.num_job_ops[i])
                    flag += 1
                elif flag == flag_new_op:
                    idx_op += 1
                    idx_ma = 0
                    flag_new_op += self.num_op_mas[self.job_op_start_idx[i] + idx_op] * 2 + 1
                    # 第二位数字：工序可选机器数
                    line_time.append(self.num_op_mas[self.job_op_start_idx[i] + idx_op])
                    line_power.append(self.num_op_mas[self.job_op_start_idx[i] + idx_op])
                    flag += 1
                elif flag_ma == 1:
                    # 第三位数字：机器编号
                    line_time.append(self.op_ma_idx[self.op_ma_start_idx[self.job_op_start_idx[i] + idx_op] + idx_ma])
                    line_power.append(self.op_ma_idx[self.op_ma_start_idx[self.job_op_start_idx[i] + idx_op] + idx_ma])
                    flag += 1
                    flag_ma = 0
                else:
                    # 第四位数字：加工时间
                    line_time.append(self.proc_time[self.op_ma_start_idx[self.job_op_start_idx[i] + idx_op] + idx_ma])
                    # 第四位数字：加工功率
                    line_power.append(self.proc_power[self.op_ma_start_idx[self.job_op_start_idx[i] + idx_op] + idx_ma])
                    flag += 1
                    flag_ma = 1
                    sum_handled_ma += 1
                    idx_ma += 1
                if sum_handled_ma == sum_ma:
                    # 各数字添加空格
                    str_line_time = " ".join([str(val_time) for val_time in line_time])
                    # 各数字添加空格
                    str_line_power = " ".join([str(val_power) for val_power in line_power])
                    # 换行并进入下一个工件
                    lines_time.append(str_line_time + '\n')
                    # 换行并进入下一个工件
                    lines_power.append(str_line_power + '\n')
                    lines_time_doc.append(str_line_time)
                    lines_power_doc.append(str_line_power)
                    break
        lines_time.append('\n')  # 各实例间换行
        lines_power.append('\n')
        if self.flag_doc:
            # 生成文件
            doc_time = open(
                self.path + '{0}j_{1}m_{2}.t'.format(self.num_jobs, self.num_mas, str.zfill(str(idx + 1), 2)), 'a')
            doc_power = open(
                self.path + '{0}j_{1}m_{2}.p'.format(self.num_jobs, self.num_mas, str.zfill(str(idx + 1), 2)), 'a')
            for i in range(len(lines_time_doc)):
                # print默认会在每个字符串末尾添加一个换行符
                print(lines_time_doc[i], file=doc_time)
            for i in range(len(lines_power_doc)):
                # print默认会在每个字符串末尾添加一个换行符
                print(lines_power_doc[i], file=doc_power)
            doc_time.close()
            doc_power.close()

        return lines_time, lines_power


if __name__ == '__main__':
    nums_ope = [configs.num_mas for _ in range(configs.num_jobs)]
    instance = InstanceGenerator(configs.num_jobs, configs.num_mas, num_job_ops=nums_ope, flag_same_op=True,
                                 path='../data_dev/0303/', flag_doc=True)
    # instance = InstanceGenerator(configs.num_jobs, configs.num_mas, path='../data_dev/2505/', flag_doc=True)
    for i in range(100):
        instance.get_instance(i)
