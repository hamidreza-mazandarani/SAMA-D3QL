import logging
from copy import copy
from importlib import reload

import numpy as np
import torch
from configs.config import experiment_configs
from utils.general_utils import make_dir


class Config:
    def __init__(self, experiment_type=None, config_profile=None, logging_enabled=True):
        """

        :param experiment_type: each experiment explores one aspect of the problem
         (e.g., number of nodes,etc.)
        :param config_profile: explores different config variables
        """
        self.experiment_type = experiment_type
        self.experiment_name = self.__load_custom_values('exp_name')

        self.config_profile = config_profile
        self.config_name = self.__load_custom_values('config_name')

        self.logging_enabled = logging_enabled

        self.folder_name = f'exp_{self.experiment_type}_profile_{self.config_profile}'
        make_dir(self.folder_name, parent='results')

        if self.logging_enabled:
            reload(logging)
            logging.basicConfig(filename=f'results/{self.folder_name}.log',
                                format='%(message)s',
                                filemode='w')
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)

        '''
        General config ************************************************************
        '''
        self.env_name = "WCNC2024_v1.1"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_iterations = self.__load_custom_values('max_iterations')
        self.num_channels = self.__load_custom_values('num_channels')
        self.number_of_executions = 1
        self.sliding_window = 1000

        # alpha in alpha-fairness
        self.alpha = self.__load_custom_values('alpha')

        # if 0, there will be much less output files, but some plots cannot be made
        # level 2 is only for running on Desktop (too much output!)
        self.verbosity_level = 1

        # if True, all algorithms in each execution use common seed
        self.use_common_seed_per_execution = True

        self.max_num_nodes = self.__load_custom_values('max_num_nodes')
        self.agents_indices = list(range(self.max_num_nodes))

        self.a_dict = self.__load_custom_values('a_dict')
        self.sparsity = self.__load_custom_values('sparsity')

        '''
        Agents config **************************************************************
        '''
        # current version is only compatible with max_packet_length == 1
        self.max_packet_length = 1
        self.header_length = 0

        self.history_length = 4

        # safety mechanism
        self.carrier_sense_enabled = False

        # actions structure:
        # [sense on channel 0, send packet with length 1 on channel 0, ..., send packet with length r on channel 0, ...
        # sense on channel 1, send packet with length 1 on channel 1, ..., send packet with length r on channel 1, ...]
        self.num_actions_per_channel = 1 + self.max_packet_length
        self.num_actions = self.num_actions_per_channel * self.num_channels

        # experience tuple: (observation, action, channel, self D2LT, assisted transmission (only for SA algorithm))
        self.num_features = len(['B', 'I', 'S', 'C']) + self.num_actions_per_channel + self.num_channels + 1 + 1

        # observation dimensions for each agent
        self.observation_shape = (self.history_length, self.num_features)

        '''
        Wireless Channels config ************************************************************
        '''
        self.context_transitions_type = None
        self.channel_transitions_type = None

        # not implemented yet

        '''
        Learning config ***********************************************************
        '''
        self.pretrained_model = False
        self.save_models_enabled = False
        self.default_training_interval = 1
        self.capacity = 500
        self.batch_size = 32
        self.lstm_state_size = 64
        self.fc_sizes = [64, 32]
        self.learning_rate = 1e-3
        self.replace_target_interval = 50
        self.gamma = 0.9

        # exploration parameters
        self.epsilon_init = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.005

    def __load_custom_values(self, attribute):
        if attribute in experiment_configs[self.experiment_type]:
            # attribute is common for all configs
            return experiment_configs[self.experiment_type][attribute]
        else:
            # per config attribute
            return experiment_configs[self.experiment_type]['configs'][self.config_profile][attribute]

    def make_banner(self, algorithm, exe_ctr):
        info = f"\nExperiment: {self.experiment_type}: {self.experiment_name} " + \
               ((57 - len(self.experiment_name)) * "-") + "\n" \
               + f"Config:     {self.config_name} " + ((60 - len(self.config_name)) * "-") + "\n" \
               + f"Algorithm:  {algorithm} " + ((60 - len(algorithm)) * "-") + "\n" \
               + f"Round:      {exe_ctr} " + ((60 - len(str(exe_ctr))) * "-") + "\n"

        return info
