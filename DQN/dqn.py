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
STATE_LEN = 4
TARGET_REPLACE_ITER = 2
MEMORY_CAPACITY = int(1e+5)
GAMMA = 0.99

'''Training settings'''
# check GPU usage
USE_GPU = torch.cuda.is_available()
print('USE GPU: ' + str(USE_GPU))
BATCH_SIZE = 64
LR = 2e-4
N_ACTIONS = 9
N_STATE = 4
N_TILE = 20


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.feature_extraction = nn.Sequential(
            # Conv2d(input channels, output channels, kernel_size, stride)
            nn.Conv2d(STATE_LEN, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(144, 512),
            nn.ReLU(),
            nn.Linear(512, 8 * 8 * 8 + N_TILE * N_STATE),
            nn.ReLU(),
            nn.Linear(8 * 8 * 8 + N_TILE * N_STATE, N_ACTIONS)
        )

        # initialization    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, state):
        if next(self.parameters()).is_cuda:
            x = x.cuda()
            state = state.cuda()

        x = x[:, -STATE_LEN:, :, :]  # extract image input; shape:(1,4,42,42)
        state = state[:, -N_STATE:]  # Extract angular velocity input; shape:(1, 4, 1)
        x = self.feature_extraction(x / 255.0)
        x = x.view(x.size(0), -1)   # shape: (1, 64)
        state = state.view(state.size(0), -1)
        state = torch.tile(state, (1, N_TILE))  # shape: (1, 80)
        x = torch.cat((x, state), 1)
        action_value = self.fc(x)

        return action_value

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        self.update_target(self.target_net, self.pred_net, 1.0)
        if USE_GPU:
            self.pred_net.cuda()
            self.target_net.cuda()

        self.memory_counter = 0
        self.learn_step_counter = 0
        self.loss_function = nn.MSELoss()
        self.replay_buffer = ReplayBufferImage(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)

    def update_target(self, target, pred, update_rate):
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) * target_param.data + update_rate * pred_param.data)

    def save_model(self, pred_path, target_path):
        self.pred_net.save(pred_path)
        self.target_net.save(target_path)

    def load_model(self, pred_path, target_path):
        self.pred_net.load(pred_path)
        self.target_net.load(target_path)

    def save_buffer(self, buffer_path):
        self.replay_buffer.save_data(buffer_path)
        print("Successfully save buffer!")

    def load_buffer(self, buffer_path):
        self.replay_buffer.read_list(buffer_path)

    # The function choose_action() should return an array
    # with a length equal to the number of environments(N_ENVS)
    def choose_action(self, x, EPSILON, idling):
        feature_component = x[0][0]
        state_component = x[0][1]

        state_tensor = torch.FloatTensor(np.array(state_component))
        feature_tensor = torch.FloatTensor(np.array(feature_component))
        state_tensor = state_tensor.view(-1, state_tensor.size(0))  # Transpose dimensions
        state_tensor = torch.tile(state_tensor, (1, N_TILE))
        state_tensor = state_tensor.transpose(0, 1)  # Restore original dimensions

        # Check if GPU usage is enabled via the 'idling' flag
        if idling:  # Assuming 'idling' indicates whether to use GPU
            feature_tensor = feature_tensor.cuda()
            state_tensor = state_tensor.cuda()

        # Implement epsilon-greedy policy
        if np.random.rand() >= EPSILON:
            feature_tensor = feature_tensor.unsqueeze(0)
            state_tensor = state_tensor.unsqueeze(0)
            action_values = self.pred_net(feature_tensor, state_tensor)     # shape:(1, 9)
            action = torch.argmax(action_values, dim=1).data.cpu().numpy()
            # print(f"Q(s,a) values: {action_values.detach().cpu().numpy()}")  # Output Q(s,a) values for all actions
        else:
            action = np.random.randint(0, N_ACTIONS, size=(feature_tensor.size(0),))

        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 1e-2)

        self.learn_step_counter += 1

        # Sample from replay buffer and convert to tensors
        state_images, state_angular_velocities, actions, rewards, next_state_images, next_state_angular_velocities, dones = self.replay_buffer.sample(
            BATCH_SIZE)
        state_images = torch.FloatTensor(state_images)
        state_angular_velocities = torch.FloatTensor(state_angular_velocities)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_state_images = torch.FloatTensor(next_state_images)
        next_state_angular_velocities = torch.FloatTensor(next_state_angular_velocities)
        dones = torch.FloatTensor(dones)

        if USE_GPU:
            state_images, state_angular_velocities, actions, rewards, next_state_images, next_state_angular_velocities, dones = \
                state_images.cuda(), state_angular_velocities.cuda(), actions.cuda(), rewards.cuda(), \
                next_state_images.cuda(), next_state_angular_velocities.cuda(), dones.cuda()

        q_eval = self.pred_net(state_images, state_angular_velocities).gather(1, actions.unsqueeze(1)).squeeze(1)
        _, a_prime = self.pred_net(next_state_images, next_state_angular_velocities).max(1)
        q_next = self.target_net(next_state_images, next_state_angular_velocities).gather(1,a_prime.unsqueeze(1)).squeeze(1)
        q_target = rewards + GAMMA * (1 - dones) * q_next

        # Calculate loss and update network
        loss = self.loss_function(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
