from abc import ABC

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from utils.general_utils import set_attributes

th.cuda.empty_cache()


class ReplayBuffer(ABC):

    def __init__(self, config):
        super().__init__()
        set_attributes(self, config)

        self.state_memory = np.zeros((self.capacity, self.max_num_nodes, self.history_length, self.num_features),
                                     dtype=np.float32)
        self.next_state_memory = np.zeros((self.capacity, self.max_num_nodes, self.history_length, self.num_features),
                                          dtype=np.float32)
        self.action_memory = np.zeros((self.capacity, self.max_num_nodes), dtype=np.int64)
        self.reward_memory = np.zeros((self.capacity, self.max_num_nodes), dtype=np.float32)
        self.terminal_memory = np.zeros((self.capacity, self.max_num_nodes), dtype=bool)

        self.mem_counter = 0

    def store_experience(self, state, next_state, action, reward, done):
        index = self.mem_counter % self.capacity

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample_buffer(self):
        max_mem = min(self.mem_counter, self.capacity)

        # "replace=False" assures that no repetitive memory is selected in batch
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = th.tensor(self.state_memory[batch]).to(self.device)
        next_state = th.tensor(self.next_state_memory[batch]).to(self.device)
        actions = th.tensor(self.action_memory[batch]).to(self.device)
        rewards = th.tensor(self.reward_memory[batch]).to(self.device)
        terminal = th.tensor(self.terminal_memory[batch]).to(self.device)

        experience = [states, next_state, actions, rewards, terminal]

        return experience

    def flush(self):
        self.state_memory = np.zeros((self.capacity, self.max_num_nodes, self.history_length, self.num_features),
                                     dtype=np.float32)
        self.next_state_memory = np.zeros((self.capacity, self.max_num_nodes, self.history_length, self.num_features),
                                          dtype=np.float32)
        self.action_memory = np.zeros((self.capacity, self.max_num_nodes), dtype=np.int64)
        self.reward_memory = np.zeros((self.capacity, self.max_num_nodes), dtype=np.float32)
        self.terminal_memory = np.zeros((self.capacity, self.max_num_nodes), dtype=bool)

        self.mem_counter = 0

    def get_all_data(self):
        return self.state_memory, self.next_state_memory, \
            self.action_memory, self.reward_memory, self.terminal_memory, \
            self.mem_counter

    def set_all_data(self, state_memory, next_state_memory, action_memory, reward_memory, terminal_memory, mem_counter):
        self.state_memory = state_memory
        self.next_state_memory = next_state_memory
        self.action_memory = action_memory
        self.reward_memory = reward_memory
        self.terminal_memory = terminal_memory
        self.mem_counter = mem_counter


class DeepQNetwork(nn.Module):
    # Reference: https://github.com/mshokrnezhad/Dueling_for_DRL

    def __init__(self, config, feature_size, name):
        nn.Module.__init__(self)
        self.feature_size = feature_size
        self.name = name
        set_attributes(self, config)

        # Build the Modules (LSTM + FC)
        self.lstm = nn.LSTM(self.feature_size, self.lstm_state_size, batch_first=True)
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(self.lstm_state_size, self.fc_sizes[0])
        self.fc_2 = nn.Linear(self.fc_sizes[0], self.fc_sizes[1])

        self.V = nn.Linear(self.fc_sizes[1], 1)
        self.A = nn.Linear(self.fc_sizes[1], self.num_actions)

        # self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate,
                                    amsgrad=True, weight_decay=0.001)
        self.loss = nn.MSELoss()

        self.to(self.device)  # move whole model to device

        # to avoid memory issues
        self.lstm.flatten_parameters()

    def forward(self, state):
        # forward propagation includes defining layers

        # remove assisted D2LT from the state if algorithm is not ours
        state = state[:, :, :self.feature_size]

        features, _ = self.lstm(state)
        x = self.relu(features[:, -1, :])

        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))

        V = self.V(x)
        A = self.A(x)

        return V, A

    def save_checkpoint(self, checkpoint_file):
        th.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(th.load(checkpoint_file))


class D3QL:
    # Reference: https://github.com/mshokrnezhad/Dueling_for_DRL

    def __init__(self, config, name, algorithm):
        self.name = name
        self.algorithm = algorithm
        set_attributes(self, config)

        self.loss = nn.MSELoss()

        # create one model per agent
        self.models = np.empty(self.max_num_nodes, dtype=object)
        self.target_models = np.empty(self.max_num_nodes, dtype=object)
        self.models_initial_weights = np.empty(self.max_num_nodes, dtype=object)

        # remove assisted D2LT from input if algorithm is not ours
        feature_size = self.num_features if self.algorithm == 'SA_plus_MA_D3QL' else (self.num_features - 1)

        for i in range(self.max_num_nodes):
            self.models[i] = DeepQNetwork(config, feature_size, name=f'{self.name}_model_{i}')
            self.target_models[i] = DeepQNetwork(config, feature_size, name=f'{self.name}_target_model_{i}')

            if self.pretrained_model:
                path = f'checkpoint/{self.folder_name}_algo_{self.algorithm}'
                self.models[i].load_checkpoint(path)

            self.models_initial_weights[i] = self.models[i].state_dict()
            self.target_models[i].load_state_dict(self.models_initial_weights[i])

        # used for updating target networks
        self.learn_step_counter = 0

        self.save_q_history = False
        self.q_history = []

        self.indexes = np.arange(self.batch_size)

    def __convert_raw_action_to_action_channel_pair(self, action):
        # convert 1-D actions (i.e. output of DQN) into 2-D actions,
        # in which a[x,c] = (action x, channel c)
        action_per_channel = action % self.num_actions_per_channel
        channel = action // self.num_actions_per_channel

        return [action_per_channel, channel]

    def __replace_target_networks(self, model_index):
        if self.learn_step_counter == 0 \
                or (self.learn_step_counter % self.replace_target_interval) == 0:
            self.target_models[model_index].load_state_dict(self.models[model_index].state_dict())

    @staticmethod
    def __convert_value_advantage_to_q_values(v, a):
        return th.add(v, (a - a.mean(dim=1, keepdim=True)))

    def get_action(self, observation, i):
        observation = th.tensor(observation, dtype=th.float).to(self.device).unsqueeze(0)
        value, advantages = self.models[i].forward(observation)

        if self.save_q_history:
            self.q_history.append([i, self.__convert_value_advantage_to_q_values(value, advantages).detach().cpu()])

        action = th.argmax(advantages).item()

        return self.__convert_raw_action_to_action_channel_pair(action)

    def train(self, states, next_states, actions, reward, dones):

        q_predicted = th.zeros((self.batch_size, self.max_num_nodes)).to(self.device)
        q_next = th.zeros((self.batch_size, self.max_num_nodes)).to(self.device)

        for i in range(self.max_num_nodes):
            # initialize local models
            self.models[i].train()
            self.models[i].optimizer.zero_grad()
            self.__replace_target_networks(model_index=i)

            V_states, A_states = self.models[i].forward(states[:, i, :, :])
            q_predicted[:, i] \
                = self.__convert_value_advantage_to_q_values(V_states, A_states)[self.indexes, actions[:, i]]

            _, A_next_states = self.models[i].forward(next_states[:, i, :, :])
            actions_states_best = A_next_states.argmax(axis=1).detach()

            V_next_states, A_next_states = self.target_models[i].forward(next_states[:, i, :, :])
            q_next_all_actions = self.__convert_value_advantage_to_q_values(V_next_states, A_next_states)
            q_next[:, i] = q_next_all_actions.gather(1, actions_states_best.unsqueeze(1)).squeeze()
            q_next[dones[:, i], i] = 0.0

        q_total = th.sum(q_predicted, dim=1, keepdim=True)
        q_total_next = th.sum(q_next, dim=1, keepdim=True)

        total_target = reward.mean(axis=-1).unsqueeze(-1) + (self.gamma * q_total_next)

        loss = self.loss(q_total, total_target).to(self.device)
        loss.backward()

        for i in range(self.max_num_nodes):
            self.models[i].optimizer.step()
            self.models[i].eval()

        self.learn_step_counter += 1

        return loss.detach().cpu().numpy()

    def get_weights(self):
        return self.model.state_dict(), self.target_model.state_dict()

    def set_weights(self, weights, weights_target):
        self.model.load_state_dict(weights)
        self.target_model.load_state_dict(weights_target)

        self.model.lstm.flatten_parameters()
        self.target_model.lstm.flatten_parameters()

    def reset_models(self):
        ...
