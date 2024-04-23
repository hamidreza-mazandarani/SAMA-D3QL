from abc import ABC

import numpy as np

from utils.general_utils import *


class WirelessChannels(ABC):
    # inspired by: The DLMA protocol (https://github.com/YidingYu/DLMA)
    # extended to multichannel scenario

    def __init__(self, config):
        super(WirelessChannels, self).__init__()
        set_attributes(self, config)

        self.counters = np.zeros((self.num_channels, self.max_num_nodes), dtype=int)
        self.success_counters = np.zeros((self.num_channels, self.max_num_nodes), dtype=int)

        self.channel_context_id = np.zeros(self.num_channels, dtype=int)
        self.channel_is_good = np.ones(self.num_channels, dtype=int)

    def observe(self, actions, channels, durations):

        actions_values = np.array(list(actions.values()))
        channel_values = np.array(list(channels.values()))

        observations_str = {}
        successful_sent_size = {}

        for idx in self.agents_indices:
            observations_str[idx], successful_sent_size[idx] = self.observe_per_node(idx,
                                                                                     durations[idx],
                                                                                     actions_values,
                                                                                     channel_values)

        return observations_str, successful_sent_size

    def observe_per_node(self, this_node_index, this_node_duration, actions, channels):
        """
        extracts observation and successful_sent_size (per channel) for one node
        """
        successful_sent_size_per_channel = np.zeros(self.num_channels)
        observation = None

        this_node_channel = channels[this_node_index]
        actions_in_same_channel = actions * (channels == this_node_channel).astype('int')
        is_transmitting_in_same_channel = np.where(actions_in_same_channel, 1, 0)
        others_transmitting_sum = is_transmitting_in_same_channel.sum() \
                                  - is_transmitting_in_same_channel[this_node_index]

        if actions[this_node_index] >= 1:
            self.counters[this_node_channel, this_node_index] += 1
            if (others_transmitting_sum == 0) and self.channel_is_good[this_node_channel]:
                self.success_counters[this_node_channel, this_node_index] += 1  # only this node ...
                # is transmitting in this time-slot and time-slot state is good

        if actions[this_node_index] > 1:
            observation = 'U'

        elif actions[this_node_index] == 1:
            if self.counters[this_node_channel, this_node_index] == this_node_duration:
                if self.success_counters[this_node_channel, this_node_index] == this_node_duration:
                    successful_sent_size_per_channel[this_node_channel] = this_node_duration - self.header_length
                    observation = 'S'
                elif self.success_counters[this_node_channel, this_node_index] < this_node_duration:
                    observation = 'C'
                else:
                    raise ValueError('success_counters cannot be larger than the duration')
                self.counters[this_node_channel, this_node_index] = 0
                self.success_counters[this_node_channel, this_node_index] = 0
            else:
                raise ValueError('counters are malfunctioning!')

        elif actions[this_node_index] == 0:
            assert self.counters[this_node_channel, this_node_index] == 0
            assert self.success_counters[this_node_channel, this_node_index] == 0

            if others_transmitting_sum == 0:  # no one is transmitting in this time-slot
                observation = 'I'
            elif others_transmitting_sum > 0:  # channel is busy by other nodes (cannot detect bad channel)
                observation = 'B'

        else:
            raise ValueError('actions cannot be negative!')

        return observation, successful_sent_size_per_channel

    def transition_context(self):
        # not implemented (only context zero)
        ...
        return self.channel_context_id

    def transition_channel(self):
        # not implemented (channel is always perfect)
        ...
        return self.channel_is_good
