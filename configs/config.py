# Topo 0: no shared information among four UEs (Bit-Oriented Network)
# Topo 1: one shared segment among two of four users
# Topo 2: complete sharing each two of four users
associations_scenario_1 = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                           [[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
                           [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
                           ]

# Topo 0: no shared information among six UEs (Bit-Oriented Network)
# Topo 1: two shared segment among each pair of six users
# Topo 2: three shared segment among each half of six users
# Topo 3: one super-node that has all segments of others
associations_scenario_2 = [
    [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],

    [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0],
     [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]],

    [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]],

    [[1, 1, 1, 1, 1],
     [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
]

alpha_values = [0, 0.5, 1, 2]
num_channel_values = [2, 3, 4, 5]
associations_sparsity_values = [1, 2, 3, 4]

# experiment configs
experiment_configs = {

    # ------------------------------------------------------------------------------------------
    # Constant associations scenarios
    # ------------------------------------------------------------------------------------------

    0: {'exp_name': 'no_transition_associations_variable_alpha_topo1',
        'max_iterations': 10000,
        'max_num_nodes': len(associations_scenario_1[1]),
        'num_channels': 1,
        'a_dict': {0: associations_scenario_1[1]},
        'sparsity': None,
        'configs':
            {idx: {'config_name': f'alpha: {alpha}',
                   'alpha': alpha,
                   } for idx, alpha in enumerate(alpha_values)},
        },

    1: {'exp_name': 'no_transition_associations_variable_alpha_topo2',
        'max_iterations': 10000,
        'max_num_nodes': len(associations_scenario_1[2]),
        'num_channels': 1,
        'a_dict': {0: associations_scenario_1[2]},
        'sparsity': None,
        'configs':
            {idx: {'config_name': f'alpha: {alpha}',
                   'alpha': alpha,
                   } for idx, alpha in enumerate(alpha_values)},
        },

    # ------------------------------------------------------------------------------------------
    # Extended associations scenarios
    # ------------------------------------------------------------------------------------------

    2: {'exp_name': 'extended_associations_variable_alpha_topo_1',
        'max_iterations': 10000,
        'max_num_nodes': len(associations_scenario_2[1]),
        'num_channels': 3,
        'a_dict': {0: associations_scenario_2[1]},
        'sparsity': None,
        'configs':
            {idx: {'config_name': f'alpha: {alpha}',
                   'alpha': alpha,
                   } for idx, alpha in enumerate(alpha_values)},
        },

    3: {'exp_name': 'extended_associations_variable_alpha_topo_2',
        'max_iterations': 10000,
        'max_num_nodes': len(associations_scenario_2[2]),
        'num_channels': 3,
        'a_dict': {0: associations_scenario_2[2]},
        'sparsity': None,
        'configs':
            {idx: {'config_name': f'alpha: {alpha}',
                   'alpha': alpha,
                   } for idx, alpha in enumerate(alpha_values)},
        },

    4: {'exp_name': 'extended_associations_variable_alpha_topo_3',
        'max_iterations': 10000,
        'max_num_nodes': len(associations_scenario_2[3]),
        'num_channels': 3,
        'a_dict': {0: associations_scenario_2[3]},
        'sparsity': None,
        'configs':
            {idx: {'config_name': f'alpha: {alpha}',
                   'alpha': alpha,
                   } for idx, alpha in enumerate(alpha_values)},
        },

}
