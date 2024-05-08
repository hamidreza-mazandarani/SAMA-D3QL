from abc import ABC
from collections import deque

import pandas as pd
from utils.env_utils import *
from utils.general_utils import *
from wireless_channels import WirelessChannels

np.seterr(divide='ignore', invalid='ignore')


def get_numpy_from_dict_values(x):
    return np.array(list(x.values()))


class Environment(ABC):

    def __init__(self, config):
        super(Environment, self).__init__()

        # set all required attributes from config file
        set_attributes(self, config)

        self.wireless_channels = WirelessChannels(config)

        # keeps last {self.history_length} pairs of (observation, action, channel)
        self.observation_vectors = {idx: deque(np.zeros((self.history_length, self.num_features)),  # single experience
                                               maxlen=self.history_length)
                                    for idx in self.agents_indices}

        # human-readable equivalent to {self.observation_vectors}, for debugging purposes
        self.observation_strings = {idx: deque(maxlen=self.history_length)
                                    for idx in self.agents_indices}

        # logs of simulation
        self.actions_arr = self.__create_per_agent_log_arrays()
        self.durations_arr = self.__create_per_agent_log_arrays()
        self.channels_arr = self.__create_per_agent_log_arrays()
        self.observations_arr = self.__create_per_agent_log_arrays(fill_value='')
        self.successful_sent_arr = self.__create_per_agent_log_arrays()
        self.rewards_arr = self.__create_per_agent_log_arrays()
        self.channel_states_df = pd.DataFrame(index=range(self.max_iterations),
                                              columns=['channel', 'channel_context'])

        # state variables
        self.self_throughputs = np.zeros((self.max_iterations, self.max_num_nodes))
        self.assisted_throughputs = np.zeros((self.max_iterations, self.max_num_nodes))
        self.total_throughputs = np.zeros((self.max_iterations, self.max_num_nodes))
        self.objective = np.zeros(self.max_iterations)

        # Delay to Last successful Transmission (D2LT) variables
        self.d2lt = np.zeros(self.max_num_nodes)
        self.d2lt_normalized = np.zeros(self.max_num_nodes)

        self.associations = None
        self.num_segments = None
        self.assistance_scores = None

    def reset(self, actions, channels):

        self.observation_vectors = {idx: deque(np.zeros((self.history_length, self.num_features)),
                                               maxlen=self.history_length)
                                    for idx in self.agents_indices}

        self.observation_strings = {idx: deque(maxlen=self.history_length)
                                    for idx in self.agents_indices}

        # logs of simulation
        self.actions_arr = self.__create_per_agent_log_arrays()
        self.durations_arr = self.__create_per_agent_log_arrays()
        self.channels_arr = self.__create_per_agent_log_arrays()
        self.observations_arr = self.__create_per_agent_log_arrays(fill_value='')
        self.successful_sent_arr = self.__create_per_agent_log_arrays()
        self.rewards_arr = self.__create_per_agent_log_arrays()
        self.channel_states_df = pd.DataFrame(index=range(self.max_iterations),
                                              columns=['channel', 'channel_context'])

        observations = {idx: 'I' for idx in self.agents_indices}
        self.__update_observation_vectors(observations, actions, channels,
                                          assisted_sent=np.zeros(self.max_num_nodes))

        return convert_queue_to_numpy(self.observation_vectors)

    def step(self, iteration, action_durations):

        # extract (action per channel, channel, duration) for each node
        actions_per_channel = {k: v[0] for k, v in action_durations.items()}
        channels = {k: v[1] for k, v in action_durations.items()}
        durations = {k: v[2] for k, v in action_durations.items()}

        # perform actions and see what happens!
        observations_str, successful_sent_size_per_channel = self.wireless_channels.observe(actions_per_channel,
                                                                                            channels, durations)
        successful_sent_size = {k: v.sum() for k, v in successful_sent_size_per_channel.items()}

        self_sent = get_numpy_from_dict_values(successful_sent_size)
        assisted_sent = calculate_assisted_transmissions(self_sent, self.associations)

        # termination conditions
        dones = {idx: False for idx in self.agents_indices}

        # used for passing extra information out of environment
        infos = {idx: {
            'observation_string': observations_str[idx],
            'successful_sent_size': successful_sent_size[idx]
        }
            for idx in self.agents_indices}

        self.__calculate_throughputs(iteration, self_sent, assisted_sent)

        rewards = {idx: (self.__calculate_rewards(iteration, idx, observations_str, successful_sent_size))
                   for idx in self.agents_indices}

        self.d2lt = np.array([0 if v > 0 else (self.d2lt[k] + 1) for k, v in successful_sent_size.items()])
        self.d2lt_normalized = self.d2lt / self.d2lt.sum()

        self.__update_stats(iteration, actions_per_channel, channels, durations, observations_str,
                            successful_sent_size, rewards)

        self.__update_observation_vectors(observations_str, durations, channels, assisted_sent)

        return convert_queue_to_numpy(self.observation_vectors), rewards, dones, infos

    def __create_per_agent_log_arrays(self, fill_value=np.nan):
        return np.full((self.max_iterations, self.max_num_nodes), fill_value)

    def __update_observation_vectors(self, observations, durations, channels, assisted_sent):
        # append (observation, action, channel) tuple, to the history queue of intelligent nodes
        # {observation_vectors} should be agents' property, but for simplicity we put it here in the environment
        for idx in observations.keys():
            if observations[idx] == 'U':
                # only update observations for agents that their actions have finished
                continue

            # original_action == duration if the node has sent a packet, otherwise 0
            original_action = durations[idx] if (observations[idx] in ['S', 'C']) else 0

            obs_vec = return_observation_vector(observations[idx])
            action_vec = return_one_hot_vector(original_action, max_value=self.num_actions_per_channel)

            channel_vec = return_one_hot_vector(channels[idx], max_value=self.num_channels)

            aggregated_obs = np.hstack([np.array(obs_vec), np.array(action_vec), np.array(channel_vec),
                                        np.array(self.d2lt_normalized[idx]),
                                        np.array(assisted_sent[idx])])
            aggregated_obs_str = [observations[idx], original_action, channels[idx]]

            self.observation_vectors[idx].append(aggregated_obs)
            self.observation_strings[idx].append(aggregated_obs_str)

    def __update_stats(self, iteration, actions, channels, durations, observations, successful_sent_size, rewards):
        self.actions_arr[iteration, :] = get_numpy_from_dict_values(actions)
        self.channels_arr[iteration, :] = get_numpy_from_dict_values(channels)
        self.durations_arr[iteration, :] = get_numpy_from_dict_values(durations)
        self.observations_arr[iteration, :] = get_numpy_from_dict_values(observations)
        self.successful_sent_arr[iteration, :] = get_numpy_from_dict_values(successful_sent_size)
        self.rewards_arr[iteration, :] = get_numpy_from_dict_values(rewards)

    def __calculate_rewards(self, iteration, idx, observations_str, successful_sent_size):

        if observations_str[idx] == 'U':
            return None

        elif observations_str[idx] == 'S':
            return successful_sent_size[idx] * self.d2lt_normalized[idx]
            # return successful_sent_size[idx] * self.assistance_scores[idx]

        elif observations_str[idx] == 'C':
            return 0

        elif observations_str[idx] in ['B', 'I']:
            # reward for observing the channel
            return 0

    def __calculate_throughputs(self, iteration, self_sent, assisted_sent):
        # moving average is implemented from scratch to reduce computations
        if iteration < self.max_packet_length:
            return
        sliding_window = min(iteration, self.sliding_window)

        # coefficients of moving average
        coefficients = (1 / sliding_window, ((sliding_window - 1) / sliding_window))

        self.self_throughputs[iteration, :] = (coefficients[0] * self_sent) \
                                              + (coefficients[1] * self.self_throughputs[iteration - 1, :])

        self.assisted_throughputs[iteration, :] = (coefficients[0] * assisted_sent) \
                                                  + (coefficients[1] * self.assisted_throughputs[iteration - 1, :])

        self.total_throughputs[iteration, :] = (coefficients[0] * (self_sent + assisted_sent)) \
                                               + (coefficients[1] * self.total_throughputs[iteration - 1, :])

        self.objective[iteration] = proportional_fairness(self.total_throughputs[iteration, :], alpha=self.alpha)

    def set_a(self, associations):
        self.associations = np.array(associations)
        self.num_segments = self.associations.shape[1]
        self.assistance_scores = (np.tile(self.associations.sum(axis=0), (self.max_num_nodes, 1))
                                  * self.associations).mean(axis=1)

    def get_variable(self, variable):
        if variable == 'total_throughputs':
            # round to decrease the size of file
            return np.round(getattr(self, variable), 2)

        return getattr(self, variable)

    @staticmethod
    def is_objective_not_stable():
        is_not_stable = True
        return is_not_stable
