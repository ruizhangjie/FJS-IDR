import argparse

parser = argparse.ArgumentParser(description='Arguments for drefjsp')
# 环境参数
parser.add_argument('--num_jobs', type=int, default=20, help='Number of jobs of instance')
parser.add_argument('--num_mas', type=int, default=10, help='Number of machines of instance')
parser.add_argument('--batch_size', type=int, default=20, help='Number of parallel instances during training')
parser.add_argument('--weight_makespan', type=float, default=0.4, help='The weight of makespan in the reward function')
parser.add_argument('--weight_energy', type=float, default=0.2,
                    help='The weight of energy consumption in the reward function')
parser.add_argument('--weight_cost', type=float, default=0.2, help='The weight of  energy cost in the reward function')
parser.add_argument('--weight_demand', type=float, default=0.2,
                    help='The weight of demand power in the reward function')

# 电价参数
parser.add_argument('--low_start_time', type=list, default=[0, 23], help='The start times of low electric price')
parser.add_argument('--low_end_time', type=list, default=[8, 24], help='The end times of low electric price')
parser.add_argument('--high_start_time', type=list, default=[9, 17], help='The start times of high electric price')
parser.add_argument('--high_end_time', type=list, default=[12, 22], help='The end times of high electric price')
parser.add_argument('--low_ele_price', type=float, default=0.2554, help='low electric price')
parser.add_argument('--mid_ele_price', type=float, default=0.5862, help='mid electric price')
parser.add_argument('--high_ele_price', type=float, default=1.0435, help='high electric price')

# 算例生成参数
parser.add_argument('--proctime_per_op_min', type=float, default=0.5, help='Minimum average processing time')
parser.add_argument('--proctime_per_op_max', type=float, default=3.0, help='Maximum average processing time')
parser.add_argument('--procpower_per_op_min', type=float, default=32.6, help='Minimum average processing power')
parser.add_argument('--procpower_per_op_max', type=float, default=97.0, help='Maximum average processing power')
parser.add_argument('--stanpower_min', type=int, default=8, help='Minimum standby power')
parser.add_argument('--stanpower_max', type=int, default=26, help='Maximum standby power')
parser.add_argument('--ops_per_job_dev', type=float, default=0.2,
                    help='Relaxation factor for the number of operations per job')
parser.add_argument('--proctime_dev', type=float, default=0.2, help='Relaxation factor for processing time sampling')
parser.add_argument('--procpower_dev', type=float, default=0.2, help='Relaxation factor for processing power sampling')

# 模型参数
parser.add_argument('--in_size_ma', type=int, default=9, help='Dimension of the raw feature vectors of machine nodes')
parser.add_argument('--out_size_ma', type=int, default=10, help='Dimension of the embedding of machine nodes')
parser.add_argument('--in_size_op', type=int, default=15,
                    help='Dimension of the raw feature vectors of operation nodes')
parser.add_argument('--out_size_op', type=int, default=10, help='Dimension of the embedding of operation nodes')
parser.add_argument('--in_size_edge', type=int, default=3, help='Dimension of the raw feature vectors of edges')
parser.add_argument('--out_size_edge', type=int, default=10, help='Dimension of the embedding of edges')
parser.add_argument('--hidden_size_op', type=int, default=128, help='Hidden dimensions of the MLPs')
parser.add_argument('--hidden_size_actor', type=int, default=128, help='Hidden dimensions of the actor')
parser.add_argument('--hidden_size_critic', type=int, default=128, help='Hidden dimensions of the critic')
parser.add_argument('--n_hidden_actor', type=int, default=3, help='Number of layers in actor')
parser.add_argument('--n_hidden_critic', type=int, default=3, help='Number of layers in critic')
parser.add_argument('--action_dim', type=int, default=1, help='Output dimension of actor')
parser.add_argument('--num_layers', type=int, default=3, help='number of hgnn layers')

# 训练参数
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--betas', type=list, default=[0.9, 0.999], help='default value for Adam')
parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
parser.add_argument('--eps_clip', type=float, default=0.2, help='clip ratio for PPO')
parser.add_argument('--K_epochs', type=int, default=4, help='update policy for K epochs')
parser.add_argument('--p_coeff', type=float, default=1.0, help='coefficient for policy loss')
parser.add_argument('--v_coeff', type=float, default=0.5, help='coefficient for value loss')
parser.add_argument('--e_coeff', type=float, default=0.01, help='coefficient for entropy term')
parser.add_argument('--minibatch_size', type=int, default=256, help='batch size for updating')
parser.add_argument('--update_iter', type=int, default=1, help='update frequency')
parser.add_argument('--max_iterations', type=int, default=3000,
                    help='maximum number of iterations for each size instance')
parser.add_argument('--ops_change_iter', type=int, default=20, help='training instance update frequency')
parser.add_argument('--save_iter', type=int, default=10, help='save frequency')
parser.add_argument('--valid_batch_size', type=int, default=10, help='Number of parallel instances during validating')

# 测试参数
parser.add_argument('--data_path', type=str, default="realcase", help='instance path')
parser.add_argument('--num_repeat', type=int, default=1000, help='the number of repeat instance')

configs = parser.parse_args()
