import numpy as np
import torch as th


def return_observation_vector(observation_string):
    if observation_string == 'B':
        return [0, 0, 0, 1]
    elif observation_string == 'I':
        return [0, 0, 1, 0]
    elif observation_string == 'S':
        return [0, 1, 0, 0]
    elif observation_string == 'C':
        return [1, 0, 0, 0]
    else:
        raise ValueError('Observation is not valid')


def return_observation_string(observation_vector):
    non_zero_index = np.nonzero(observation_vector)[0][0]
    if non_zero_index == 4:
        return 'B'
    elif non_zero_index == 3:
        return 'I'
    elif non_zero_index == 2:
        return 'S'
    elif non_zero_index == 1:
        return 'C'
    elif non_zero_index == 0:
        return 'U'
    else:
        raise ValueError('Vector format is not valid')


def return_one_hot_vector(value, max_value):
    one_hot_vector = [0 for _ in range(max_value)]
    one_hot_vector[value] = 1
    return one_hot_vector


def convert_queue_to_numpy(queue):
    return {k: np.vstack([x for x in v]) for k, v in queue.items()}


def proportional_fairness(x, alpha=0.1):
    if alpha == 0:
        return x.sum()
    elif alpha == 1:
        return th.log(x).sum() if th.is_tensor(x) else np.log(x).sum()
    elif (0 < alpha < 1) or (1 < alpha < 100):
        return ((x ** (1 - alpha)) / (1 - alpha)).sum()
    else:
        raise ValueError('alpha must be non-negative and less than 100')


def calculate_assisted_transmissions(self_sent, associations):
    num_segments = associations.shape[1]
    self_sent_per_segment = np.tile(np.expand_dims(self_sent, 1), (1, num_segments)) * associations
    sent_segments = self_sent_per_segment.max(axis=0)
    total_sent = (sent_segments * associations).sum(axis=1) / associations.sum(axis=1)
    assisted_sent = total_sent - self_sent

    return assisted_sent
