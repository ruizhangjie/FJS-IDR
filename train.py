import os
import random
import time
from collections import deque

import numpy as np
import torch

from PPO_Model.PPO import Memory, PPO
from env.DREFJSP_Env import DREFJSP
from env.instance_generator import InstanceGenerator
from params import configs
from validate import get_validate_env, validate


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None,
                           sci_mode=False)

    # 加载配置
    num_jobs = configs.num_jobs
    num_mas = configs.num_mas
    batch_size = configs.batch_size
    ops_per_job_min = int(num_mas * (1 - configs.ops_per_job_dev))  # The minimum number of operations for a job
    ops_per_job_max = int(num_mas * (1 + configs.ops_per_job_dev))

    # 初始化环境
    memories = Memory()
    model = PPO(batch_size)
    env_valid = get_validate_env()  # Create an environment for validation
    mul_obj_best = float('inf')
    reward_best = float('-inf')
    sample_parm = 0
    maxlen = 1  # Save the best model
    best_models = deque()

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './record/train_{0}'.format(str_time)
    os.makedirs(save_path)

    # 训练迭代
    start_time = time.time()
    env = None
    step_rewards = []
    eqal_targets = []
    step_loss = []

    # 相同大小的算例迭代3000次
    for i in range(1, configs.max_iterations + 1):
        if (i - 1) % configs.ops_change_iter == 0:
            # 每10次重新生成一次算例
            nums_ope = [random.randint(ops_per_job_min, ops_per_job_max) for _ in range(num_jobs)]
            instance = InstanceGenerator(num_jobs, num_mas, num_job_ops=nums_ope, flag_same_op=True)
            env = DREFJSP(instance)
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))
        state = env.state
        done = False
        last_time = time.time()

        # 并行调度
        while ~done:
            # 禁用梯度计算
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, sample_parm)
            state, rewards, dones = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
        print("spend_time: ", time.time() - last_time)
        env.reset()

        # 按照配置的频率更新策略
        if i % configs.update_iter == 0:
            loss, reward = model.update(memories)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            step_rewards.append(reward)
            step_loss.append(loss)
            memories.clear_memory()
            if reward >= reward_best:
                sample_parm = sigmoid(reward).item()
                reward_best = reward
            else:
                sample_parm = 0

        if i % configs.save_iter == 0:
            print('\nStart validating')
            # Record the average results and the results on each instance
            value = validate(env_valid, model.policy_old)
            eqal_targets.append(value)
            # Save the best model
            if value < mul_obj_best:
                mul_obj_best = value
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

    print("total_time: ", time.time() - start_time)
    np.savetxt('{0}/rewards.txt'.format(save_path), np.array(step_rewards), fmt='%.4f')
    np.savetxt('{0}/targets.txt'.format(save_path), np.array(eqal_targets), fmt='%.4f')
    np.savetxt('{0}/loss.txt'.format(save_path), np.array(step_loss), fmt='%.4f')


def sigmoid(z):
    a = torch.tensor([-z])
    return 1 / (1 + torch.exp(a))


if __name__ == '__main__':
    main()
