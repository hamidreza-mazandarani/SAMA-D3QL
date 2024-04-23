import logging

import torch

from configs.config import experiment_configs
from configs.config_maker import Config
from simulation_runner import SimulationRunner
from utils.general_utils import *

# see config.py for definitions of experiments
experiment_types = [2, 3, 4]

# set 1 to run the algorithm, otherwise 0
algorithms = {'Random': 1, 'MA_D3QL': 1, 'SA_MA_D3QL': 0, 'SA_plus_MA_D3QL': 1}

for experiment_type in experiment_types:

    # get the list of configs for this experiment
    configs_list = list(experiment_configs[experiment_type]['configs'].keys())

    for config_profile in configs_list:

        # create a config file
        config = Config(experiment_type, config_profile)

        seeds = np.random.randint(0, int(1e5), config.number_of_executions)
        config.logger.debug(f'seeds are: {seeds}')

        a_matrices_all = [create_stochastic_associations(config.max_num_nodes,
                                                         num_segments_per_user=config.sparsity)
                          for _ in range(config.number_of_executions)]

        for algorithm, algorithm_is_enabled in algorithms.items():

            if not algorithm_is_enabled:
                continue

            # loop over execution rounds
            for exe_ctr in range(config.number_of_executions):

                # initialize seeds
                if config.use_common_seed_per_execution:
                    torch.manual_seed(seeds[exe_ctr])
                    np.random.seed(seeds[exe_ctr])

                if config.a_dict is None:
                    config.a_dict = {0: a_matrices_all[exe_ctr]}
                    config.logger.debug(f'Matrices are: {config.a_dict}')

                info = config.make_banner(algorithm, exe_ctr)
                print(info)
                config.logger.debug(info)
                runner = SimulationRunner(config, algorithm)
                runner.run_one_episode()

                if config.verbosity_level == 0:
                    runner.log_episode_stats(execution_counter=exe_ctr,
                                             stats=('observations_arr', 'total_throughputs'))
                elif config.verbosity_level == 1:
                    runner.log_episode_stats(execution_counter=exe_ctr)
                else:
                    runner.log_episode_stats(execution_counter=exe_ctr)
                    runner.save_sbs_as_file(execution_counter=exe_ctr)

                if config.save_models_enabled:
                    runner.save_models()

                del runner
                config.logger.debug(f'\n\n\n\n\n')

        logging.shutdown()
