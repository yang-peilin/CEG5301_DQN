###########################################################################################
# Implementation of Deep Q-Learning Networks (DQN)
# Paper: https://www.nature.com/articles/nature14236
# Reference: https://github.com/Kchu/DeepRL_PyTorch
###########################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from replay_memory import ReplayBufferImage


'''DQN settings'''
# sequential images to define state
# 强化学习代理在决策时会考虑最近的 4 帧图像
STATE_LEN = 4
# target policy sync interval
# 每进行 2 次训练迭代就同步一次目标网络的权重
TARGET_REPLACE_ITER = 2
# (prioritized) experience replay memory size
# 经验回放 (Experience Replay) 缓冲区用于存储代理的过去经历（状态、动作、奖励、下一状态），并在训练时从中随机采样
MEMORY_CAPACITY = int(1e+5)
# gamma for MDP
# 马尔可夫决策过程（MDP）的折扣因子 (discount factor)
GAMMA = 0.99

'''Training settings'''
# check GPU usage
USE_GPU = torch.cuda.is_available()
print('USE GPU: '+str(USE_GPU))
# mini-batch size
BATCH_SIZE = 64
# learning rate
LR = 2e-4
# the number of actions 
N_ACTIONS = 9
# 环境中代理可以选择的动作有 9 种不同的可能性
# the dimension of states
N_STATE = 4
# 每个状态由 4 个特征组成
# the multiple of tiling states
# 用来“覆盖状态空间”的倍数或数量
N_TILE = 20


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.feature_extraction = nn.Sequential(
            # Conv2d(input channels, output channels, kernel_size, stride)
            nn.Conv2d(STATE_LEN, 8, kernel_size=8, stride=2),
            
            # TODO: ADD SUITABLE CNN LAYERS TO ACHIEVE BETTER PERFORMANCE
        )
           
        # action value
        self.fc_q = nn.Linear(8 * 8 * 8 + N_TILE * N_STATE, N_ACTIONS) 
        
        # TODO: ADD SUITABLE FULLY CONNECTED LAYERS TO ACHIEVE BETTER PERFORMANCE
        
        # initialization    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
    def forward(self, x, state):
        # x.size(0) : minibatch size
        mb_size = x.size(0)
        x = self.feature_extraction(x / 255.0) # (m, 9 * 9 * 10)
        x = x.view(x.size(0), -1)
        state = state.view(state.size(0), -1)
        state = torch.tile(state, (1, N_TILE))
        x = torch.cat((x, state), 1)
        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))
        action_value = self.fc_q(x)

        return action_value

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        

class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        # sync target net
        self.update_target(self.target_net, self.pred_net, 1.0)
        # use gpu
        if USE_GPU:
            self.pred_net.cuda()
            self.target_net.cuda()
            
        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0
        # loss function
        self.loss_function = nn.MSELoss()
        # create the replay buffer
        self.replay_buffer = ReplayBufferImage(MEMORY_CAPACITY)
        
        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)
        
    def update_target(self, target, pred, update_rate):
        # update target network parameters using prediction network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate * pred_param.data)
    
    def save_model(self, pred_path, target_path):
        # save prediction network and target network
        self.pred_net.save(pred_path)
        self.target_net.save(target_path)

    def load_model(self, pred_path, target_path):
        # load prediction network and target network
        self.pred_net.load(pred_path)
        self.target_net.load(target_path)
        
    def save_buffer(self, buffer_path):
        self.replay_buffer.save_data(buffer_path)
        print("Successfully save buffer!")

    def load_buffer(self, buffer_path):
        # load data from the pkl file
        self.replay_buffer.read_list(buffer_path)

    def choose_action(self, s, epsilon, idling):
        
        # TODO: REPLACE THE FOLLOWING PLACEHOLDER CODE WITH YOUR CODE

        # NOTE: If you enabled multiple threading (N_ENVS number of environments), then you should sample N_ENVS number of
        # actions that matches the number of environments. E.g.,
        # action = np.random.randint(0, N_ACTIONS, N_ENVS)
        # where N_ENVS can be inferred from the appropriate dimensions of s (which stacks the states from all N_ENVS env's).
        # For example:
        image = np.stack([item[0] for item in s])
        N_ENVS = image.shape[0]
        action = np.random.randint(0, N_ACTIONS, N_ENVS)
        
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
    
        # TODO: REPLACE THE FOLLOWING PLACEHOLDER CODE WITH YOUR CODE
        
        loss = 1
        
        return loss
