import math
import random

import numpy as np
import torch
from matplotlib import pyplot as plt, rcParams


def colour_gen(n):
    '''
    为工件生成随机颜色
    :param n: 工件数
    :return: 颜色列表
    '''
    color_bits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    colours = []
    random.seed(234)
    for i in range(n):
        colour_bits = ['#']
        colour_bits.extend(random.sample(color_bits, 6))
        colours.append(''.join(colour_bits))
    return colours


def initialize_plt(num_mas, num_jobs, ma_idx, ma_start, ma_end, start_op, path, num):
    y_value = list(range(1, num_mas + 1))
    max_time = 0
    # plt.figure(figsize=(12, 4))
    plt.figure(figsize=(10, 2))
    # plt.xlabel('time', size=14, fontdict={'family': 'Times New Roman'})
    # plt.ylabel('machine', size=14, fontdict={'family': 'Times New Roman'})
    plt.yticks(y_value, fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)

    colors = colour_gen(num_jobs)
    # 设置字体属性
    # font = {'family': 'Times New Roman',
    #         'color': 'black',
    #         'weight': 'normal',
    #         'size': 6,
    #         }
    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'normal',
            'size': 10,
            }

    for i in range(ma_end.shape[0]):
        end_idx = torch.nonzero(ma_end[i] > 0).squeeze()
        if len(end_idx)>0:
            end_idx = end_idx[-1]
            for k in range(end_idx + 1):
                # job号
                possible_pos = torch.where(ma_idx[i, k] < start_op)[0]
                if (len(possible_pos) == 0):
                    job = torch.tensor(num_jobs)
                    op = ma_idx[i, k] - start_op[-1] + 1
                else:
                    job = possible_pos[0]
                    op = ma_idx[i, k] - start_op[possible_pos[0] - 1] + 1
                start_time = ma_start[i, k]
                if ma_end[i, k] > max_time:
                    max_time = ma_end[i, k]
                dur = ma_end[i, k] - ma_start[i, k]
                plt.barh(i + 1, dur.item(), 0.5, left=start_time.item(), color=colors[job - 1])
                plt.text(start_time.item(), i + 1.3, '%s/%s' % (job.item(), op.item()), fontdict=font)

    # 获取当前的Axes对象
    ax = plt.gca()
    # 设置x轴的范围，使其起始位置为0,跨度为5
    max_time = find_multiple(max_time, 2)
    x = generate_sequence(max_time, 2)
    # 设置x轴刻度的位置
    ax.set_xticks(x)
    # 设置x轴刻度的标签
    ax.set_xticklabels(x)
    # 隐藏上方和右侧的边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # 只显示左侧和底部的刻度
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    # 设置刻度线朝内
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')

    plt.savefig("%s/J%sM%s_%s_gantt.jpg" % (path, num_jobs, num_mas, num), dpi=600, bbox_inches='tight')
    plt.close()
    return max_time


def generate_sequence(max_value, step, flag_gannt=True):
    sequence = []
    a = 0
    while True:
        # 根据给定的规则生成元素
        if a == 0:
            value = 0
        else:
            if flag_gannt:
                value = step * a
            else:
                value = step * a - 1

        # 如果生成的元素大于最大值，跳出循环
        if value > max_value:
            break

        # 将生成的元素添加到序列中
        sequence.append(value)

        # 增加索引
        a += 1

    return sequence


def find_multiple(number, num):
    # 找到比给定数大的最小整数
    ceil_number = math.ceil(number)

    # 如果这个数是5的倍数，直接返回
    if ceil_number % num == 0:
        return ceil_number
    else:
        # 找到最近的大于这个数的5的倍数
        return ceil_number + num - ceil_number % num


def plt_demand(num_mas, num_jobs, ma_start, ma_end, ma_power, ma_sb_start, ma_sb_end, ma_sb_power, path, num, max_time):
    # 准备数据
    ma_sb_power = ma_sb_power.unsqueeze(-1).expand(-1, ma_start.shape[1])
    ma_start = torch.where(ma_start == -99., 0, ma_start)
    ma_power = torch.where(ma_power == -99., 0, ma_power)
    ma_sb_power = torch.where(ma_sb_start == -99., 0, ma_sb_power)
    ma_sb_start = torch.where(ma_sb_start == -99., 0, ma_sb_start)

    # 处理数据
    pro_power = cal_demand(ma_power, ma_start, ma_end, max_time)
    sb_power = cal_demand(ma_sb_power, ma_sb_start, ma_sb_end, max_time)
    total_power = pro_power + sb_power
    y = total_power.shape[1]

    # 绘制
    rcParams['font.family'] = 'Times New Roman'
    # 创建一个用于堆叠图的数组
    stacked_data = np.cumsum(total_power, axis=0)
    # 创建x轴的数据
    x = np.arange(y)
    # 创建堆叠面积图
    # fig, ax = plt.subplots(figsize=(12, 3))
    fig, ax = plt.subplots(figsize=(10, 2))
    color1 = [244 / 255, 111 / 255, 68 / 255]
    color2 = [127 / 255, 203 / 255, 164 / 255]
    color3 = [75 / 255, 101 / 255, 175 / 255]
    color = []
    color.append(color1)
    color.append(color2)
    color.append(color3)
    ax.fill_between(x, 0, stacked_data[0, :], color=color[0])
    for i in range(1, total_power.shape[0]):
        ax.fill_between(x, stacked_data[i - 1, :], stacked_data[i, :], color=color[i])
    ax.plot(x, stacked_data[-1, :], color='red', linewidth=2)
    # 设置x轴的范围，使其起始位置为0
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim([0, y])
    xticks = generate_sequence(y, 20, False)
    xlbls = generate_sequence(y // 10, 2)
    # 设置x轴刻度的位置
    ax.set_xticks(xticks)
    # 设置x轴刻度的标签
    ax.set_xticklabels(xlbls)
    # 隐藏上方和右侧的边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # 只显示左侧和底部的刻度
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    # 设置刻度线朝内
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')

    plt.savefig("%s/J%sM%s_%s_load.jpg" % (path, num_jobs, num_mas, num), dpi=600, bbox_inches='tight')
    plt.close()


def cal_demand(power, start_time, end_time, max_time):
    max = max_time * 10
    a = start_time.unsqueeze(0).expand(max, -1, -1)
    b = end_time.unsqueeze(0).expand(max, -1, -1)
    c = power.unsqueeze(0).expand(max, -1, -1)
    steps = torch.arange(0, max_time, step=0.1, dtype=torch.float32)
    d = steps.unsqueeze(-1).unsqueeze(-1).expand(-1, a.shape[1], a.shape[2])
    e = torch.where(a <= d, d, torch.tensor(-1.))
    e = torch.where(e < b, e, torch.tensor(-1.))
    f = torch.where(e != -1, torch.tensor(1.), torch.tensor(0.))
    g = f * c
    result = torch.sum(g, dim=-1).t()
    return result.cpu().numpy()
