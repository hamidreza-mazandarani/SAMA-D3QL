from abc import ABC

import numpy as np

from model import ReplayBuffer, D3QL
from utils.general_utils import *


def duration(a):
    return max(1, a)


class SBS(ABC):
    def __init__(self, config, algorithm):
        super().__init__()
        set_attributes(self, config)
        self.config = config
        self.algorithm = algorithm

        self.d3ql_algorithm = D3QL(self.config, name=f'sbs', algorithm=self.algorithm)

        self.buffer = ReplayBuffer(self.config)

        self.loss_history = []

    def add_aggregated_experience_to_buffer(self, iteration, experience=None):
        if not experience:
            return

        old_observation = np.array([v for k, v in experience[0].items()])
        new_observation = np.array([v for k, v in experience[1].items()])
        observation_string = np.array([v['observation_string'] for k, v in experience[2].items()])
        action_durations = np.array([v for k, v in experience[3].items()])
        rewards = np.array([v for k, v in experience[4].items()])
        done = np.array([v for k, v in experience[5].items()])

        action_1d = np.zeros(self.max_num_nodes)
        for i in range(self.max_num_nodes):
            action_per_channel = action_durations[i, 2] \
                if (observation_string[i] in ['S', 'C']) else 0
            channel = action_durations[i, 1]
            action_1d[i] = action_per_channel + (self.num_actions_per_channel * channel)

        self.buffer.store_experience(
            old_observation, new_observation, action_1d, rewards, done)

    def is_training_required(self, iteration, extra_condition=True):
        return ((iteration % self.default_training_interval) == 0) \
            and (self.algorithm != 'Random') and extra_condition

    def train_on_random_samples(self):
        if self.buffer.mem_counter < self.batch_size:
            return

        sample_experience = self.buffer.sample_buffer()
        loss = self.d3ql_algorithm.train(*sample_experience)

        self.loss_history.append(loss)

    def manage_contexts(self, iteration, a):
        ...
