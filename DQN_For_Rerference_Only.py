###########################################################################################
# Implementation of Deep Q-Learning Networks (DQN)
# Author for codes: Chu Kun(kun_chu@outlook.com)
# Paper: https://www.nature.com/articles/nature14236
# Reference: https://github.com/sungyubkim/Deep_RL_with_pytorch
###########################################################################################
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay_memory import ReplayBuffer, PrioritizedReplayBuffer

import random
import os
import pickle
import time
from collections import deque
import matplotlib.pyplot as plt
from wrappers import wrap, wrap_cover, SubprocVecEnv

import argparse
parser = argparse.ArgumentParser(description='Some settings of the experiment.')
parser.add_argument('games', type=str, nargs=1, help='name of the games. for example: Breakout')
args = parser.parse_args()
args.games = "".join(args.games)

'''DQN settings'''
# sequential images to define state
STATE_LEN = 4
# target policy sync interval
TARGET_REPLACE_ITER = 1
# simulator steps for start learning
LEARN_START = int(1e+3)
# (prioritized) experience replay memory size
MEMORY_CAPACITY = int(1e+5)
# simulator steps for learning interval
LEARN_FREQ = 4

'''Environment Settings'''
# number of environments for DQN
N_ENVS = 16
# Total simulation step
STEP_NUM = int(1e+8)
# gamma for MDP
GAMMA = 0.99
# visualize for agent playing
RENDERING = False
# openai gym env name
ENV_NAME = args.games+'NoFrameskip-v4'
env = SubprocVecEnv([wrap_cover(ENV_NAME) for i in range(N_ENVS)])
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape

'''Training settings'''
# check GPU usage
USE_GPU = torch.cuda.is_available()
print('USE GPU: '+str(USE_GPU))
# mini-batch size
BATCH_SIZE = 64
# learning rage
LR = 1e-4
# epsilon-greedy
EPSILON = 1.0

'''Save&Load Settings'''
# check save/load
SAVE = True
LOAD = False
# save frequency
SAVE_FREQ = int(1e+3)
# paths for predction net, target net, result log
PRED_PATH = './data/model/dqn_pred_net_o_'+args.games+'.pkl'
TARGET_PATH = './data/model/dqn_target_net_o_'+args.games+'.pkl'
RESULT_PATH = './data/plots/dqn_result_o_'+args.games+'.pkl'

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.feature_extraction = nn.Sequential(
        	# Conv2d(Input channels, Output channels, kernel_size, stride)
            # 输入通道数为STATE_LEN，即输入状态的帧数
            nn.Conv2d(STATE_LEN, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(7 * 7 * 64, 512)
        
        # action value
        self.fc_q = nn.Linear(512, N_ACTIONS) 
        
        # Initialization
        # 初始化卷积神经网络（CNN）中的各层参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                # 使用Xavier正态分布（Xavier Normal Initialization）来初始化卷积层的权重
                # Xavier 初始化是为了保持每一层的输入和输出方差一致
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # 使用 Kaiming 正态初始化（Kaiming Normal Initialization）来初始化全连接层的权重
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
    def forward(self, x):
        # x.size(0) : minibatch size
        mb_size = x.size(0)
        x = self.feature_extraction(x / 255.0) # (m, 7 * 7 * 64)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))                 
        action_value = self.fc_q(x)

        # 输出每个动作的Q值
        return action_value


    def save(self, PATH):
        torch.save(self.state_dict(),PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))

class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        # sync evac target
        # 更新目标网络（target_net），确保它和预测网络（pred_net）之间的参数保持一致或者部分一致
        self.update_target(self.target_net, self.pred_net, 1.0)
        # use gpu
        if USE_GPU:
            self.pred_net.cuda()
            self.target_net.cuda()
            
        # simulator step counter
        # 通常用于记录在训练过程中经历的步骤数
        self.memory_counter = 0
        # target network step counter
        # 跟踪目标网络（targetnetwork）更新的步数（也就是训练中调用 learn() 方法的次数）
        self.learn_step_counter = 0
        # loss function
        self.loss_function = nn.MSELoss()
        # ceate the replay buffer
        # 创建一个经验回放缓冲区（ReplayBuffer），用于存储智能体与环境交互的经验，并在后续训练中通过采样这些经验来更新神经网络的参数
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
        
        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)

    # 将预测网络（predictionnetwork）的参数部分或全部复制到目标网络（targetnetwork）中
    def update_target(self, target, pred, update_rate):
        # update target network parameters using predcition network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate*pred_param.data)
    
    def save_model(self):
        # save prediction network and target network
        self.pred_net.save(PRED_PATH)
        self.target_net.save(TARGET_PATH)

    def load_model(self):
        # load prediction network and target network
        self.pred_net.load(PRED_PATH)
        self.target_net.load(TARGET_PATH)

    def choose_action(self, x, EPSILON):
    	# x:state
        x = torch.FloatTensor(x)
        # print(x.shape)
        if USE_GPU:
            x = x.cuda()

        # epsilon-greedy policy
        if np.random.uniform() >= EPSILON:
            # greedy case
            action_value = self.pred_net(x) 	# (N_ENVS, N_ACTIONS, N_QUANT)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
        else:
            # random exploration case
            action = np.random.randint(0, N_ACTIONS, (x.size(0)))
        return action

    # 用于将智能体与环境的交互经历（即“状态 - 动作 - 奖励 - 下一状态”）
    # 存储到经验回放缓冲区（Replay Buffer）
    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
        self.learn_step_counter += 1
        # target parameter update
        # 在一定的学习步骤后，将预测网络的参数更新到目标网络中
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 1e-2)

        # 从经验回放缓冲区（Replay Buffer）中抽取一个批次（mini - batch）的经验样本
        b_s, b_a, b_r, b_s_, b_d = self.replay_buffer.sample(BATCH_SIZE)
        # b_w, b_idxes = np.ones_like(b_r), None
            
        b_s = torch.FloatTensor(b_s)
        b_a = torch.LongTensor(b_a)
        b_r = torch.FloatTensor(b_r)
        b_s_ = torch.FloatTensor(b_s_)
        b_d = torch.FloatTensor(b_d)

        if USE_GPU:
            b_s, b_a, b_r, b_s_, b_d = b_s.cuda(), b_a.cuda(), b_r.cuda(), b_s_.cuda(), b_d.cuda()

        # action value for current state
        # 通过 预测网络（prediction network）计算所有动作的 Q 值
        q_eval = self.pred_net(b_s)
        # 获取批次的大小
        mb_size = q_eval.size(0)
        # 选择每个样本对应的特定动作的Q值
        q_eval = torch.stack([q_eval[i][b_a[i]] for i in range(mb_size)])

        # optimal action value for current state
        # 计算每个样本的目标Q值（q_target），用于更新深度Q网络（DQN）的预测网络参数
        q_next = self.target_net(b_s_) 				
        # best_actions = q_next.argmax(dim=1) 		
        # q_next = torch.stack([q_next[i][best_actions[i]] for i in range(mb_size)])
        # 获取每个下一状态的最大Q值
        q_next = torch.max(q_next, -1)[0]
        # 计算目标Q值
        q_target = b_r + GAMMA * (1. - b_d) * q_next
        # 从计算图中分离目标Q值
        q_target = q_target.detach()

        # loss
        loss = self.loss_function(q_eval, q_target)
        
        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

dqn = DQN()

# model load with check
if LOAD and os.path.isfile(PRED_PATH) and os.path.isfile(TARGET_PATH):
    dqn.load_model()
    pkl_file = open(RESULT_PATH,'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
    print('Load complete!')
else:
    result = []
    print('Initialize results!')

print('Collecting experience...')

# episode step for accumulate reward 
epinfobuf = deque(maxlen=100)
# check learning time
start_time = time.time()

# env reset
s = np.array(env.reset())
# print(s.shape)

# for step in tqdm(range(1, STEP_NUM//N_ENVS+1)):
# STEP_NUM：通常表示总的训练步数
for step in range(1, STEP_NUM//N_ENVS+1):
    a = dqn.choose_action(s, EPSILON)
    # print('a',a)

    # take action and get next state
    s_, r, done, infos = env.step(a)
    # log arrange
    for info in infos:
        maybeepinfo = info.get('episode')
        if maybeepinfo: epinfobuf.append(maybeepinfo)
    s_ = np.array(s_)

    # clip rewards for numerical stability
    clip_r = np.sign(r)

    # store the transition
    for i in range(N_ENVS):
        dqn.store_transition(s[i], a[i], clip_r[i], s_[i], done[i])

    # annealing the epsilon(exploration strategy)
    if step <= int(1e+4):
        # linear annealing to 0.9 until million step
        EPSILON -= 0.9/1e+4
    elif step <= int(2e+4):
    # else:
        # linear annealing to 0.99 until the end
        EPSILON -= 0.09/1e+4

    # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
    if (LEARN_START <= dqn.memory_counter) and (dqn.memory_counter % LEARN_FREQ == 0):
        loss = dqn.learn()

    # print log and save
    if step % SAVE_FREQ == 0:
        # check time interval
        time_interval = round(time.time() - start_time, 2)
        # calc mean return
        mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]),2)
        result.append(mean_100_ep_return)
        # print log
        print('Used Step: ',dqn.memory_counter,
              '| EPS: ', round(EPSILON, 3),
              # '| Loss: ', loss,
              '| Mean ep 100 return: ', mean_100_ep_return,
              '| Used Time:',time_interval)
        # save model
        dqn.save_model()
        pkl_file = open(RESULT_PATH, 'wb')
        pickle.dump(np.array(result), pkl_file)
        pkl_file.close()

    s = s_

    if RENDERING:
        env.render()
print("The training is done!")