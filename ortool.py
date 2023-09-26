import collections
import time

import torch
from ortools.sat.python import cp_model

from env.TOU_Cost import electricity_cost, cal_demand
from env.load_data import nums_extraction
from params import configs
from plot_gantt import initialize_plt, plt_demand


def load_drefjs(lines_time):
    # 读取算例输出初始特征张量
    flag_line = 0
    result = []  # 输出
    # Parse data line by line
    for line in lines_time:
        # first line 首行不处理
        if flag_line == 0:
            flag_line += 1
        # last line 尾行结束
        elif line == "\n":
            break
        # other
        else:
            # Detect information of this job and return the number of operations
            line_time_split = line.split()
            flag = 0  # 指针
            flag_ma = 1  # 1-处理机器，0-处理时间/功率
            flag_new_op = 1  # 新工序指针
            idx_op = -1  # 正在处理的工序idx
            job = []
            op = []
            ma = 0
            # i代表具体数字
            for i in line_time_split:
                x = float(i)
                # The first number indicates the number of operations of this job
                if flag == 0:
                    flag += 1
                # new operation detected
                elif flag == flag_new_op:
                    if len(op) > 0:
                        job.append(op)
                        op = []
                    idx_op += 1
                    flag_new_op += x * 2 + 1
                    flag += 1
                # not proc_time (machine)
                elif flag_ma == 1:
                    ma = int(x - 1)
                    flag += 1
                    flag_ma = 0
                # proc_time
                else:
                    op.append((int(x * 10), ma))
                    flag += 1
                    flag_ma = 1
            job.append(op)
            result.append(job)
            flag_line += 1
    return result


def load_power(lines_power, allop, ma):
    # 读取算例输出初始特征张量
    flag_line = 0
    result = torch.zeros(size=(allop, ma))  # 输出
    num_job_ops = []  # A list of the number of operations for each job
    job_op_start_idx = []
    # Parse data line by line
    for line in lines_power:
        # first line 首行不处理
        if flag_line == 0:
            flag_line += 1
        # last line 尾行结束
        elif line == "\n":
            break
        # other
        else:
            op_start_idx = int(sum(num_job_ops))
            job_op_start_idx.append(op_start_idx)
            # Detect information of this job and return the number of operations
            line_power_split = line.split()
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
                    result[idx_op + op_start_idx][ma] = x
                    flag += 1
                    flag_ma = 1
            num_job_ops.append(num_ops)
            flag_line += 1
    return result, job_op_start_idx


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        """Called at each new solution."""
        print('Solution %i, time = %f s, objective = %i' %
              (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        self.__solution_count += 1


def flexible_jobshop(jobs, ma, allop, power, start_idx, stand_power):
    """Solve a small flexible jobshop problem."""

    num_jobs = len(jobs)
    all_jobs = range(num_jobs)

    num_machines = ma
    all_machines = range(num_machines)

    # Model the flexible jobshop problem.
    model = cp_model.CpModel()

    horizon = 0
    for job in jobs:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration

    print('Horizon = %i' % horizon)

    # Global storage of variables.
    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []

    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = '_j%i_t%i' % (job_id, task_id)
            start = model.NewIntVar(0, horizon, 'start' + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration,
                                       'duration' + suffix_name)
            end = model.NewIntVar(0, horizon, 'end' + suffix_name)
            interval = model.NewIntervalVar(start, duration, end,
                                            'interval' + suffix_name)

            # Store the start for the solution.
            starts[(job_id, task_id)] = start

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = '_j%i_t%i_a%i' % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar('presence' + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence,
                        'interval' + alt_suffix)
                    l_presences.append(l_presence)

                    # Link the primary/global variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Add the local interval to the right machine.
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence

                # Select exactly one presence variable.
                model.AddExactlyOne(l_presences)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)

        job_ends.append(previous_end)

    # Create machines constraints.
    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # Solve model.

    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter()
    solver.parameters.max_time_in_seconds = 3600.0
    status = solver.Solve(model, solution_printer)

    start_time = []
    end_time = []
    pro_time = []
    op_power = []
    ma_start = -1 * torch.ones(ma, allop)
    ma_end = torch.zeros(ma, allop)
    # Print final solution.
    for job_id in all_jobs:
        print('Job %i:' % job_id)
        for task_id in range(len(jobs[job_id])):
            start_value = solver.Value(starts[(job_id, task_id)])
            machine = -1
            duration = -1
            selected = -1
            for alt_id in range(len(jobs[job_id][task_id])):
                if solver.Value(presences[(job_id, task_id, alt_id)]):
                    duration = jobs[job_id][task_id][alt_id][0]
                    machine = jobs[job_id][task_id][alt_id][1]
                    selected = alt_id
            print(
                '  task_%i_%i starts at %i (alt %i, machine %i, duration %i)' %
                (job_id, task_id, start_value, selected, machine, duration))
            start_value = int(start_value) / 10.0
            machine = int(machine)
            start_time.append(start_value)
            duration = int(duration) / 10.0
            pro_time.append(duration)
            end_value = start_value + duration
            end_time.append(end_value)
            row = ma_start[machine]
            positions = torch.nonzero((row > -1) & (row < start_value)).flatten()
            if len(positions) == 0:
                row = torch.cat((torch.tensor([start_value]), row[:-1]))
            else:
                pos = positions[-1]
                row = torch.cat((row[:pos + 1], torch.tensor([start_value]), row[pos + 1:-1]))
            ma_start[machine] = row
            row1 = ma_end[machine]
            positions1 = torch.nonzero((row1 > 0) & (row1 < end_value)).flatten()
            if len(positions1) == 0:
                row1 = torch.cat((torch.tensor([end_value]), row1[:-1]))
            else:
                pos = positions1[-1]
                row1 = torch.cat((row1[:pos + 1], torch.tensor([end_value]), row1[pos + 1:-1]))
            ma_end[machine] = row1
            opnum = start_idx[job_id] + task_id
            op_power.append(power[opnum][machine])
    pro_time = torch.tensor(pro_time)
    op_power = torch.tensor(op_power)
    consumption = torch.sum(op_power * pro_time)
    start_time = torch.tensor(start_time)
    end_time = torch.tensor(end_time)
    cost = electricity_cost(op_power, start_time, end_time, torch.tensor(configs.low_start_time),
                            torch.tensor(configs.low_end_time),
                            torch.tensor(configs.high_start_time), torch.tensor(configs.high_end_time),
                            torch.tensor(configs.low_ele_price),
                            torch.tensor(configs.mid_ele_price), torch.tensor(configs.high_ele_price))
    demand = cal_demand(op_power.unsqueeze(0), start_time.unsqueeze(0), end_time.unsqueeze(0))
    # 计算待机时间
    ma_start = ma_start[:, 1:]
    # 添加全部为-1的元素到最后一列
    negative_ones_column = torch.full((ma_start.shape[0], 1), -1)
    ma_start = torch.cat((ma_start, negative_ones_column), dim=1)
    ma_start[ma_start == -1] = 0
    for row in ma_end:
        # 找到第一个0元素的索引
        zero_index = (row == 0).nonzero(as_tuple=False)
        if len(zero_index) > 0:
            zero_index = zero_index[0, 0]
            # 将第一个0元素的前一个元素变成0
            if zero_index > 0:
                row[zero_index - 1] = 0

    power1 = stand_power.unsqueeze(-1).expand(-1, allop)
    consumption1 = torch.sum(power1 * (ma_start - ma_end))
    consumption += consumption1
    start_flat = ma_start.flatten(0)
    end_flat = ma_end.flatten(0)
    power_flat = power1.flatten(0)
    cost1 = electricity_cost(power_flat, end_flat, start_flat, torch.tensor(configs.low_start_time),
                             torch.tensor(configs.low_end_time),
                             torch.tensor(configs.high_start_time), torch.tensor(configs.high_end_time),
                             torch.tensor(configs.low_ele_price),
                             torch.tensor(configs.mid_ele_price), torch.tensor(configs.high_ele_price))
    cost = torch.sum(cost1) + torch.sum(cost)
    demand1 = cal_demand(power_flat, end_flat, start_flat.unsqueeze(0))
    demand = torch.max(demand + demand1, dim=1)[0]
    makespan = int(solver.ObjectiveValue()) / 10.
    cost = cost.item()
    consumption = consumption.item()
    demand = demand.item()

    print('Solve status: %s' % solver.StatusName(status))
    print(f'Makespan:{makespan:.2f}')
    print(f'Cost:{cost:.2f}')
    print(f'Consumption:{consumption:.2f}')
    print(f'Demand:{demand:.2f}')
    print('Statistics')
    print('  - conflicts : %i' % solver.NumConflicts())
    print('  - branches  : %i' % solver.NumBranches())
    print('  - wall time : %f s' % solver.WallTime())


if __name__ == '__main__':
    lines_time = []
    lines_power = []
    path = 'data_test/0303/3j_3m_01.t'
    with open(path) as file_object:
        line = file_object.readlines()
        lines_time.append(line)
    path = 'data_test/0303/3j_3m_01.p'
    with open(path) as file_object:
        line = file_object.readlines()
        lines_power.append(line)
    num_jobs, num_mas, all_max_ops = nums_extraction(lines_time[0])
    result = load_drefjs(lines_time[0])
    power, start_idx = load_power(lines_power[0], all_max_ops, num_mas)
    standby_power_ma = torch.tensor([3, 2, 4])
    # standby_power_ma = torch.tensor([25, 8, 9, 17, 23])
    # standby_power_ma = torch.tensor([10, 25, 8, 9, 21, 18, 15, 17, 14, 23])
    # standby_power_ma = torch.tensor([10, 25, 8, 9, 21, 18, 15, 17, 14, 23, 10, 25, 8, 9, 21, 18, 15, 17, 14, 23])
    flexible_jobshop(result, num_mas, all_max_ops, power, start_idx, standby_power_ma)
    ma_start = torch.tensor([[0, 8, 20, 30], [0, 12, 30, -99], [8, 22, -99, -99]])
    ma_end = torch.tensor([[8, 20, 30, 43], [12, 26, 40, 0], [22, 37, 0, 0]])
    ma_op = torch.tensor([[0, 3, 4, 8], [6, 7, 5, -99], [1, 2, -99, -99]])
    ma_power = torch.tensor([[94, 92, 96, 97], [97, 92, 90, -99], [98, 100, -99, -99]])
    ma_sb_start = torch.tensor([[-99, -99, -99, -99], [26, -99, -99, -99], [-99, -99, -99, -99]])
    ma_sb_end = torch.tensor([[0, 0, 0, 0], [30, 0, 0, 0], [0, 0, 0, 0]])
    op_start = torch.tensor([0, 3, 6])
    max_time = initialize_plt(num_mas, num_jobs, ma_op, ma_start,
                              ma_end, op_start, 'data_test/0303', 1)
    plt_demand(num_mas, num_jobs, ma_start, ma_end, ma_power, ma_sb_start, ma_sb_end, standby_power_ma,
               'data_test/0303', 1,
               max_time)
