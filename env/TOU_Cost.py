import copy

import torch


def cal_demand(power, start_time, end_time):
    # 以7天为最大期限
    a = start_time.unsqueeze(0).expand(672, -1, -1)
    a = torch.floor(a * 4)
    b = end_time.unsqueeze(0).expand(672, -1, -1)
    b = torch.floor(b * 4)
    c = power.unsqueeze(0).expand(672, -1, -1)
    steps = torch.arange(0, 672, dtype=torch.float32)
    d = steps.unsqueeze(-1).unsqueeze(-1).expand(-1, a.shape[1], a.shape[2])
    e = torch.where(a <= d, d, torch.tensor(-1.))
    e = torch.where(e < b, e, torch.tensor(-1.))
    f = torch.where(e != -1, torch.tensor(1.), torch.tensor(0.))
    g = f * c
    result = torch.sum(g, dim=-1).t()
    return result


def electricity_cost(power, start_time, end_time, low_start, low_end, high_start, high_end, low_ele_price,
                     mid_ele_price, high_ele_price):
    dt = 0.1
    deal = False
    if start_time.dim() == 1:
        start_time = start_time.unsqueeze(0)
        end_time = end_time.unsqueeze(0)
        power = power.unsqueeze(0)
        deal = True
    s = start_time.unsqueeze(0).expand(30, -1, -1)  # 单个op最大加工时间为3,间隔取0.1
    # 计算c的元素
    steps = dt * torch.arange(0, 30, dtype=torch.float32)  # 生成0到最大步数的向量
    b = steps.unsqueeze(-1).unsqueeze(-1).expand(-1, s.shape[1], s.shape[2])
    c = b + s  # 将超过b的部分设置为b的值
    e = end_time.unsqueeze(0).expand(30, -1, -1)
    i = torch.where(e > c, c, torch.tensor(-1.))
    i = torch.fmod(i, 24)
    # 判断条件并生成新的张量m
    i = torch.where(((i.unsqueeze(-1) >= low_start) & (i.unsqueeze(-1) < low_end)).any(dim=-1), torch.tensor(-99.), i)
    i = torch.where(((i.unsqueeze(-1) >= high_start) & (i.unsqueeze(-1) < high_end)).any(dim=-1), torch.tensor(-98.), i)
    i[(i != -99) & (i != -98) & (i != -1)] = torch.tensor([mid_ele_price])
    i[i == -99] = torch.tensor([low_ele_price])
    i[i == -98] = torch.tensor([high_ele_price])
    i[i == -1] = torch.tensor(0.)
    power = power.unsqueeze(0).expand(30, -1, -1)
    result = torch.sum(power * i * 0.1, dim=0)
    if deal:
        result = result.squeeze(dim=0)
    return result

if __name__ == '__main__':
    from params import configs
    op_power = torch.tensor([3])
    start_time = torch.tensor([23.2])
    end_time = torch.tensor([24.7])
    cost = electricity_cost(op_power, start_time, end_time, torch.tensor(configs.low_start_time),
                            torch.tensor(configs.low_end_time),
                            torch.tensor(configs.high_start_time), torch.tensor(configs.high_end_time),
                            torch.tensor(configs.low_ele_price),
                            torch.tensor(configs.mid_ele_price), torch.tensor(configs.high_ele_price))
    print(cost)








