import pickle

import numpy as np
import pandas as pd
import tqdm

from agents import IntelligentAgent
from base_station import SBS
from env import Environment
from utils.general_utils import *


class SimulationRunner:

    def __init__(self, config, algorithm):
        self.config = config

        # algorithm of intelligent agents
        self.algorithm = algorithm

        # set all required attributes from config file
        set_attributes(self, config)

        # create a single environment
        self.env = Environment(self.config)

        # create the Small Base Station (SBS)
        self.sbs = SBS(self.config, algorithm=self.algorithm)

        # create agents based on their protocols
        self.agents = {}
        for idx in range(self.max_num_nodes):
            self.agents[idx] = IntelligentAgent(config=self.config, agent_index=idx, algorithm=self.algorithm)

        # create progress tracker
        self.T = tqdm.trange(self.max_iterations, desc='Progress', leave=True,
                             disable=self.verbosity_level != 2)

    def run_one_episode(self):

        # initial values of the simulation
        actions = {idx: 0 for idx in self.agents_indices}
        channels = {idx: self.agents[idx].channel for idx in self.agents_indices}
        infos = {idx: {'observation_string': 'I'} for idx in self.agents_indices}

        # reset the environment
        observations = self.env.reset(actions, channels)

        # simulation loop
        for iteration in self.T:
            self.__update_progress_bar(metrics_to_show={
                'Throughputs':
                    self.env.get_variable('total_throughputs')[iteration - 1]
            })

            # get actions from agents
            action_durations = {idx: agent.make_decision(
                actions=self.sbs.d3ql_algorithm.get_action(observations[idx], idx),
                observation_string=infos[idx]['observation_string'],
                iteration=iteration)
                for idx, agent in self.agents.items()}

            previous_observations = observations.copy()

            if self.__check_context_transition_occurred(iteration):
                if self.algorithm in ['Random', 'SA_MA_D3QL', 'SA_plus_MA_D3QL']:
                    associations = self.a_dict[iteration]
                else:
                    associations = np.eye(self.max_num_nodes)
                self.env.set_a(associations=associations)

                self.sbs.manage_contexts(iteration, a=associations)

            # pass the action_durations to the environment and get observations and other info
            observations, rewards, dones, infos = self.env.step(iteration, action_durations)

            # create an aggregated experience of agents with finished actions
            experiences = [previous_observations, observations, infos, action_durations, rewards, dones]

            self.sbs.add_aggregated_experience_to_buffer(iteration=iteration,
                                                         experience=experiences)

            if self.sbs.is_training_required(iteration):
                self.sbs.train_on_random_samples()

    def log_episode_stats(self, execution_counter,
                          stats=('actions_arr', 'observations_arr', 'successful_sent_arr',
                                 'channels_arr', 'rewards_arr',
                                 'self_throughputs', 'assisted_throughputs', 'total_throughputs', 'objective')):

        make_dir(f'algo_{self.algorithm}', parent=f'results/{self.folder_name}')

        for x in stats:
            stat_arr = self.env.get_variable(x)
            stat_df = pd.DataFrame(stat_arr)
            stat_df.to_csv(f'results/{self.folder_name}/algo_{self.algorithm}/exe_{execution_counter}_{x}.csv',
                           index=False)

    def save_sbs_as_file(self, execution_counter):

        make_dir(self.folder_name, parent='results')
        make_dir(f'algo_{self.algorithm}', parent=f'results/{self.folder_name}')

        file_name = f'results/{self.folder_name}/algo_{self.algorithm}/exe_{execution_counter}.pickle'
        with open(file_name, 'wb') as handle:
            pickle.dump(self.sbs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_models(self):

        make_dir(self.folder_name, parent='results')
        make_dir(f'algo_{self.algorithm}', parent=f'results/{self.folder_name}')

        for idx, x in self.agents.items():
            if x.protocol == 'intelligent':
                if x.algorithm != 'Random':
                    path = f'checkpoint/{self.folder_name}_algo_{self.algorithm}'
                    x.d3ql_algorithm.model.save_checkpoint(path)

    def __check_context_transition_occurred(self, iteration):
        # returns True if there is a change in the value of associations
        return iteration in self.a_dict.keys()

    def __get_active_agents_per_channel(self, iteration):
        active_agents_per_channel = [([(x.protocol, x.profile) for x in self.agents.values() if
                                       (x.activation_time_vector[iteration] == 1) and (x.channel == c)
                                       and (x.protocol != 'intelligent')])
                                     for c in range(self.num_channels)]

        return active_agents_per_channel

    def __update_progress_bar(self, metrics_to_show):
        description = ""
        for k, v in metrics_to_show.items():
            description = description.join(f"--- {k}: {v} ---")
        self.T.set_description(description)
        self.T.refresh()
