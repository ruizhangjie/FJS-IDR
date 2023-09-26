import copy
import math
import random

import torch
from torch import nn
from torch.distributions import Categorical

from PPO_Model.mlp import MLPActor, MLPCritic
from disjunctive_graph.mgnn import GATop, MLPma
from params import configs
import torch.nn.functional as F


class MGNNs(nn.Module):
    def __init__(self):
        super(MGNNs, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.in_size_ma = configs.in_size_ma
        self.out_size_ma = configs.out_size_ma
        self.in_size_op = configs.in_size_op
        self.out_size_op = configs.out_size_op
        self.in_size_edge = configs.in_size_edge
        self.out_size_edge = configs.out_size_edge
        self.hidden_size_op = configs.hidden_size_op
        self.hidden_size_actor = configs.hidden_size_actor
        self.hidden_size_critic = configs.hidden_size_critic
        self.actor_dim = configs.out_size_ma * 2 + configs.out_size_op * 2
        self.critic_dim = configs.out_size_ma + configs.out_size_op
        self.n_hidden_actor = configs.n_hidden_actor
        self.n_hidden_critic = configs.n_hidden_critic
        self.action_dim = configs.action_dim
        self.num_layers = configs.num_layers

        # len() means of the number of HGNN iterations
        # and the element means the number of heads of each HGNN (=1 in final experiment)
        # Operation node embedding
        self.get_operations = nn.ModuleList()
        self.get_operations.append(GATop([self.in_size_ma+self.in_size_edge, self.in_size_op, self.in_size_op, self.in_size_op],
                                         self.hidden_size_op, self.out_size_op))
        # Machine node embedding
        self.get_machines = nn.ModuleList()
        self.get_machines.append(MLPma([self.out_size_op, self.in_size_ma], self.out_size_ma, self.hidden_size_op))
        for i in range(1, self.num_layers):
            self.get_operations.append(GATop([self.out_size_ma+self.in_size_edge, self.out_size_op, self.out_size_op, self.out_size_op],
                                             self.hidden_size_op, self.out_size_op))
            self.get_machines.append(MLPma([self.out_size_op, self.out_size_ma], self.out_size_ma, self.hidden_size_op))

        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.hidden_size_actor, self.action_dim).to(
            self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.hidden_size_critic, 1).to(self.device)

    def forward(self):
        '''
        Replaced by separate act and evaluate functions
        '''
        raise NotImplementedError

    def act(self, state, memories, sample_parm=0.7, flag_train=True):
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        action_probs, op_step_batch = self.get_action_prob(state, memories, flag_train=flag_train)

        # DRL-S, sampling actions following
        if flag_train:
            if torch.rand(1) >= sample_parm:
                dist = Categorical(action_probs)  # 可以用于生成离散分布
                action_indexes = dist.sample()  # 从分布中采样一个值
            else:
                dist = Categorical(action_probs)  # 可以用于生成离散分布
                action_indexes = action_probs.argmax(dim=1)
        # DRL-G, greedily picking actions with the maximum probability
        else:
            if torch.rand(1) >= sample_parm:
                dist = Categorical(action_probs)  # 可以用于生成离散分布
                action_indexes = dist.sample()  # 从分布中采样一个值
            else:
                action_indexes = action_probs.argmax(dim=1)
        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()
        ops = op_step_batch[state.batch_idxes, jobs]

        # Store data in memory during training
        if flag_train == True:
            memories.logprobs.append(dist.log_prob(action_indexes))
            memories.action_indexes.append(action_indexes)

        return torch.stack((jobs, ops, mas), dim=0)

    def get_action_prob(self, state, memories, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        batch_idxes = state.batch_idxes

        # Raw feats
        raw_ops = state.feat_ops_batch.transpose(1, 2)[batch_idxes]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        raw_edge = state.feat_edge_batch[batch_idxes]

        # Normalize
        nums_ops = state.sum_ops_batch[batch_idxes]
        features = self.get_normalized(raw_ops, raw_mas, raw_edge, batch_idxes, nums_ops, flag_train)
        norm_ops = (copy.deepcopy(features[0]))
        norm_mas = (copy.deepcopy(features[1]))
        norm_edge = (copy.deepcopy(features[2]))

        # L iterations of the HGNN
        for i in range(self.num_layers):
            # First Stage, operation node embedding
            # shape: [len(batch_idxes), num_ops, out_size_ope]
            h_ops = self.get_operations[i](state.op_ma_adj_batch, state.op_pre_adj_batch, state.op_sub_adj_batch,
                                           state.batch_idxes, features)
            features = (h_ops, features[1], features[2])
            # Second Stage, machine node embedding
            # shape: [len(batch_idxes), num_mas, out_size_ma]
            h_mas = self.get_machines[i](state.op_ma_adj_batch, state.batch_idxes, features)
            features = (features[0], h_mas, features[2])

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ma]
        # There may be different operations for each instance, which cannot be pooled directly by the matrix 训练时取相同ops序列
        if not flag_train:
            h_ops_pooled = []
            for i in range(len(batch_idxes)):
                h_ops_pooled.append(torch.mean(h_ops[i, :nums_ops[i], :], dim=-2))
            h_ops_pooled = torch.stack(h_ops_pooled)  # shape: [len(batch_idxes), out_size_ope]
        else:
            h_ops_pooled = h_ops.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ope]

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        start_idx_step_batch = torch.where(state.start_idx_step_batch > state.job_op_end_idx_batch,
                                           state.job_op_end_idx_batch, state.start_idx_step_batch)
        # 扩展（或重复）一个 tensor 的维度
        jobs_gather = start_idx_step_batch[..., :, None].expand(-1, -1, h_ops.size(-1))[batch_idxes]
        h_jobs = h_ops.gather(1, jobs_gather)  # 各job就绪op
        # Matrix indicating whether processing is possible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible_proc = state.op_ma_adj_batch[batch_idxes].gather(1,
                                                                  start_idx_step_batch[..., :, None].expand(-1, -1,
                                                                                                            state.op_ma_adj_batch.size(
                                                                                                                -1))[
                                                                      batch_idxes])
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.op_ma_adj_batch.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_ops_pooled_padding = h_ops_pooled[:, None, None, :].expand_as(h_jobs_padding)
        # Matrix indicating whether job is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        job_eligible = ~(state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible = job_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return

        # Input of actor MLP
        # shape: [len(batch_idxes), num_mas, num_jobs, out_size_ma*2+out_size_ope*2]
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_ops_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        mask = eligible.transpose(1, 2).flatten(1)

        # Get priority index and probability of actions with masking the ineligible actions
        scores = self.actor(h_actions).flatten(1)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)

        # Store data in memory during training
        if flag_train == True:
            memories.op_ma_adj.append(copy.deepcopy(state.op_ma_adj_batch))
            memories.op_pre_adj.append(copy.deepcopy(state.op_pre_adj_batch))
            memories.op_sub_adj.append(copy.deepcopy(state.op_sub_adj_batch))
            memories.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memories.raw_ops.append(copy.deepcopy(norm_ops))
            memories.raw_mas.append(copy.deepcopy(norm_mas))
            memories.raw_edge.append(copy.deepcopy(norm_edge))
            memories.nums_ops.append(copy.deepcopy(nums_ops))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.eligible.append(copy.deepcopy(eligible))

        return action_probs, start_idx_step_batch

    def get_normalized(self, raw_ops, raw_mas, raw_edge, batch_idxes, nums_ops, flag_train=False):
        '''
        :param raw_ops: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param raw_edge: Processing time、功率、能耗
        :param batch_idxes: Uncompleted instances
        :param nums_ops: The number of operations for each instance
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        '''
        batch_size = batch_idxes.size(0)  # number of uncompleted instances

        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        if not flag_train:
            mean_ops = []
            std_ops = []
            for i in range(batch_size):
                mean_ops.append(torch.mean(raw_ops[i, :nums_ops[i], :], dim=-2, keepdim=True))
                std_ops.append(torch.std(raw_ops[i, :nums_ops[i], :], dim=-2, keepdim=True))
                for j in range(raw_edge.size(-1)):
                    proc_idxes = torch.nonzero(raw_edge[i, :, :, j])
                    proc_values = raw_edge[i, proc_idxes[:, 0], proc_idxes[:, 1], j]
                    proc_norm = self.feature_normalize(proc_values)
                    raw_edge[i, proc_idxes[:, 0], proc_idxes[:, 1], j] = proc_norm
            mean_ops = torch.stack(mean_ops, dim=0)
            std_ops = torch.stack(std_ops, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            edge_norm = raw_edge
        # DRL-S and scheduling during training have a consistent number of operations
        else:
            mean_ops = torch.mean(raw_ops, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            std_ops = torch.std(raw_ops, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            sliced_tensors = torch.unbind(raw_edge, dim=-1)
            normalized_tensors = []
            # 循环固定次数3次
            for t in sliced_tensors:
                normalized_tensors.append(self.feature_normalize(t))
            stacked_tensor = torch.stack(normalized_tensors, dim=-1)
            edge_norm = stacked_tensor  # shape: [len(batch_idxes), num_ops, num_mas, in_size_edge]
        return ((raw_ops - mean_ops) / (std_ops + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
                edge_norm)

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    def evaluate(self, op_ma_adj, op_pre_adj, op_sub_adj, raw_ops, raw_mas, raw_edge,
                 jobs_gather, eligible, action_envs):
        batch_idxes = torch.arange(0, op_ma_adj.size(-3)).long()
        features = (raw_ops, raw_mas, raw_edge)

        # L iterations of the HGNN
        for i in range(self.num_layers):
            h_ops = self.get_operations[i](op_ma_adj, op_pre_adj, op_sub_adj, batch_idxes, features)
            features = (h_ops, features[1], features[2])
            h_mas = self.get_machines[i](op_ma_adj, batch_idxes, features)
            features = (features[0], h_mas, features[2])

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)
        h_opes_pooled = h_ops.mean(dim=-2)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        h_jobs = h_ops.gather(1, jobs_gather)
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, op_ma_adj.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        scores = self.actor(h_actions).flatten(1)
        mask = eligible.transpose(1, 2).flatten(1)

        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze().double(), dist_entropys


class Memory:
    def __init__(self):
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indexes = []

        self.op_ma_adj = []
        self.op_pre_adj = []
        self.op_sub_adj = []
        self.batch_idxes = []
        self.raw_ops = []
        self.raw_mas = []
        self.raw_edge = []
        self.jobs_gather = []
        self.eligible = []
        self.nums_ops = []

    def clear_memory(self):
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]

        del self.op_ma_adj[:]
        del self.op_pre_adj[:]
        del self.op_sub_adj[:]
        del self.batch_idxes[:]
        del self.raw_ops[:]
        del self.raw_mas[:]
        del self.raw_edge[:]
        del self.jobs_gather[:]
        del self.eligible[:]
        del self.nums_ops[:]


class PPO:
    def __init__(self, num_envs=None):
        self.lr = configs.lr
        self.betas = configs.betas
        self.gamma = configs.gamma
        self.eps_clip = configs.eps_clip
        self.K_epochs = configs.K_epochs
        self.p_coeff = configs.p_coeff
        self.v_coeff = configs.v_coeff
        self.e_coeff = configs.e_coeff
        self.num_envs = num_envs  # Number of parallel instances
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.policy = MGNNs().to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        minibatch_size = configs.minibatch_size

        # Flatten the data in memory (in the dim of parallel instances and decision points)
        old_op_ma_adj = torch.stack(memory.op_ma_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_op_pre_adj = torch.stack(memory.op_pre_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_op_sub_adj = torch.stack(memory.op_sub_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_ops = torch.stack(memory.raw_ops, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_mas = torch.stack(memory.raw_mas, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_edge = torch.stack(memory.raw_edge, dim=0).transpose(0, 1).flatten(0, 1)
        old_jobs_gather = torch.stack(memory.jobs_gather, dim=0).transpose(0, 1).flatten(0, 1)
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0, 1)
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0, 1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0, 1).flatten(0, 1)
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0, 1).flatten(0, 1)

        # Estimate and normalize the rewards
        rewards_envs = []
        discounted_rewards = 0
        for i in range(self.num_envs):
            rewards = []
            discounted_reward = 0
            # zip按索引一一配对,reversed反向遍历
            for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_is_terminals[i])):
                if is_terminal:
                    discounted_rewards += discounted_reward
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            discounted_rewards += discounted_reward
            rewards = torch.tensor(rewards, dtype=torch.float64).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_envs.append(rewards)
        rewards_envs = torch.cat(rewards_envs)

        loss_epochs = 0
        full_batch_size = old_op_ma_adj.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for i in range(num_complete_minibatches + 1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_op_ma_adj[start_idx: end_idx, :, :],
                                         old_op_pre_adj[start_idx: end_idx, :, :],
                                         old_op_sub_adj[start_idx: end_idx, :, :],
                                         old_raw_ops[start_idx: end_idx, :, :],
                                         old_raw_mas[start_idx: end_idx, :, :],
                                         old_raw_edge[start_idx: end_idx, :, :],
                                         old_jobs_gather[start_idx: end_idx, :, :],
                                         old_eligible[start_idx: end_idx, :, :],
                                         old_action_envs[start_idx: end_idx])

                ratios = torch.exp(logprobs - old_logprobs[i * minibatch_size:(i + 1) * minibatch_size].detach())
                advantages = rewards_envs[i * minibatch_size:(i + 1) * minibatch_size] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = - self.p_coeff * torch.min(surr1, surr2) \
                       + self.v_coeff * self.MseLoss(state_values,
                                                     rewards_envs[i * minibatch_size:(i + 1) * minibatch_size]) \
                       - self.e_coeff * dist_entropy
                loss_epochs += loss.mean().detach()

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_epochs.item() / self.K_epochs, \
               discounted_rewards.item() / (self.num_envs * configs.update_iter)
