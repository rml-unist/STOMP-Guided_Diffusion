import os
import ray
import isaacgym
from itertools import product

from inference_classics import experiment
# from experiment_launcher import Launcher
# from experiment_launcher.utils import is_local
# from experiment_launcher import run_experiment

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########################################################################################################################
ray.init(num_gpus=1, num_cpus=4) # Please match this number with CUDA_VISIBLE_DEVICES

@ray.remote(num_gpus=1, num_cpus=4) # Please also match this number with ray.init, do not use num_gpus < 1
def one_experiment(model_id, planner_alg, num_trajectory, seed, save_dir):
    experiment(
        model_id=model_id,

        planner_alg = planner_alg, #'RRTC', 'GPMP', 'stochGPMP'

        num_trajectories = num_trajectory, #For GPMP, stochGPMP

        debug = False,
        render = False,

        #######################################
        # MANDATORY
        seed = seed,
        save_dir = save_dir,
        #######################################

    )

########################################################################################################################
# EXPERIMENT PARAMETERS SETUP
main_dir = './'

# planner_algs = ['RRTC']
# num_trajectories = [1]

planner_algs = ['GPMP', 'stochGPMP']
num_trajectories = [50, 70, 100]

model_ids = [
    'EnvSpheres3D-RobotPanda',
    'EnvSimple2D-RobotPointMass',
    'EnvNarrowPassageDense2D-RobotPointMass',
    'EnvDense2D-RobotPointMass',
]

sample_vers = ['noise_ddpm']

seeds = list(range(0, 2))

########################################################################################################################
# RUN
futures = []
for i, (model_id, planner_alg, num_trajectory, seed, sample_ver) in enumerate(product(model_ids, planner_algs, num_trajectories, seeds, sample_vers)):
    save_dir = f'{main_dir}MPDresults{planner_alg}_{num_trajectory}/'
    future = one_experiment.remote(model_id, planner_alg, num_trajectory, seed, save_dir)
    futures.append(future)

while futures:
    done, futures = ray.wait(futures, num_returns=1) # only return one task
    ray.get(done)  # Get the result of the completed task