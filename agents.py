from abc import ABC

import numpy as np
from utils.general_utils import *


def duration(a):
    return max(1, a)


class IntelligentAgent(ABC):
    def __init__(self, config, agent_index, algorithm):
        super().__init__()
        set_attributes(self, config)

        self.agent_index = agent_index
        self.protocol = 'intelligent'
        self.algorithm = algorithm

        self.channel = 0

        self.is_active = True

        # [action_per_channel, channel_index, duration >= 1]
        # need to store durations of actions since they may last more than one step
        self.action_durations = [0, 0, duration(0)]

        # exploration variables
        self.epsilon = self.epsilon_init
        self.epsilons_log = []

        if self.algorithm != 'Random':
            self.prob_per_action = np.arange(self.num_actions_per_channel, 0, -1).astype(float)
            self.prob_per_action /= self.prob_per_action.sum()
        else:
            self.prob_per_action = None

    def make_decision(self, actions, observation_string, iteration):
        """
        determines what to do in the current iteration (i.e., time slot)
        :param actions: selected actions
        we actually get actions from the SBS, but the correct way is to copy models to the agents \
        and call the get_action function there.
        :param observation_string: current observation in string format
        :param iteration: current iteration
        :return: [action_per_channel, channel_index, duration >= 1]
        """

        if observation_string == 'U':
            # Unfinished packet transmissions (before taking a new action, the previous action must finish)
            if self.action_durations[0] < 1:
                raise ValueError('action value cannot be less than one when observation is U')
            if self.action_durations[0] > self.action_durations[2]:
                raise ValueError('action value cannot be greater than the duration')
            self.action_durations[0] -= 1
            return self.action_durations

        if not self.is_active:
            # no action
            self.action_durations = [0, 0, duration(0)]
            return self.action_durations

        if self.protocol == 'intelligent':
            self.action_durations = self.intelligent_transmit(actions, observation_string)
            return self.action_durations

    def __update_epsilon(self, reset=False):
        if not reset:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        else:
            self.epsilon = self.epsilon_init
        self.epsilons_log.append(self.epsilon)

    def intelligent_transmit(self, actions, observation_string):
        self.__update_epsilon()

        if self.carrier_sense_enabled and (observation_string != 'I'):
            last_channel = self.action_durations[1]
            return [0, last_channel, duration(0)]

        if (np.random.random() < self.epsilon) or (self.algorithm == 'Random'):
            # exploration or random algorithm
            random_channel = np.random.choice(np.arange(self.num_channels))
            random_action_per_channel = np.random.choice(np.arange(self.num_actions_per_channel),
                                                         p=self.prob_per_action)

            return [random_action_per_channel, random_channel, duration(random_action_per_channel)]
        else:
            # exploitation
            action_per_channel, channel = actions

            return [action_per_channel, channel, duration(action_per_channel)]
