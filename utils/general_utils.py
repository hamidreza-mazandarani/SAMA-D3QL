import os

import numpy as np


def set_attributes(_self, config):
    attributes = config.__dict__
    for k, v in attributes.items():
        setattr(_self, k, v)


def make_dir(folder_name, parent):
    if not (folder_name in os.listdir(parent)):
        os.makedirs(parent + '/' + folder_name)


def create_stochastic_associations(num_users, num_segments=None, num_segments_per_user=1):
    if num_segments is None:
        num_segments = num_users

    a = np.zeros((num_users, num_segments))

    for i in range(num_users):
        indices = np.random.choice(num_segments, num_segments_per_user)
        a[i, indices] = 1

    return a


def make_segment_similarity_matrix(a):
    num_users = len(a)

    w = np.zeros((num_users, num_users))

    for i in range(num_users):
        for j in range(num_users):
            if i != j:
                w[i, j] = (a[i, :] * a[j, :]).sum() / (a[j, :]).sum()

    return w
