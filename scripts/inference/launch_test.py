import os
import ray
import isaacgym
from itertools import product

from inference import experiment
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local
# from experiment_launcher import run_experiment

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

########################################################################################################################
ray.init(num_gpus=2, num_cpus=8) # Please match this number with CUDA_VISIBLE_DEVICES

@ray.remote(num_gpus=1, num_cpus=4) # Please also match this number with ray.init, do not use num_gpus < 1
def one_experiment(model_id, seed, sample_ver, n_samples, save_dir):
    sample_num=5 
    if model_id == 'EnvSpheres3D-RobotPanda':
        sample_num=15

    experiment(
        model_id=model_id,
        planner_arg='mpd',

        sample_ver=sample_ver,

        latent_guide = False,
        gradient_free_guide = True, 
        gradient_free_guide_ver = 'STOMP', 

        n_samples=n_samples,
        step_size=0, #step_size,
        negative_step_size=0, #step_size,
        n_post_gradient_step=0, #stomp_step,

        # this will disable other conditons
        resample = True,
        guidance_sample=sample_num,
        time_sample_ver='diffusion-es',

        render=False,
        seed=seed,
        save_dir=save_dir,

        debug=False,
    )

########################################################################################################################
# EXPERIMENT PARAMETERS SETUP
save_dir = './MPDresultsDiffusionES/'

model_ids = [
    'EnvSpheres3D-RobotPanda',
    # 'EnvSimple2D-RobotPointMass',
    # 'EnvNarrowPassageDense2D-RobotPointMass',
    'EnvDense2D-RobotPointMass',
]

sample_vers = ['noise_ddpm']

seeds = list(range(0, 300))
n_samples = 100

# step_sizes = [3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005]
# stomp_num_steps = [5, 10, 15]

render = False

########################################################################################################################
# RUN
futures = []
for i, (model_id, seed, sample_ver) in enumerate(product(model_ids, seeds, sample_vers)):
    future = one_experiment.remote(model_id, seed, sample_ver, n_samples, save_dir)
    futures.append(future)

while futures:
    done, futures = ray.wait(futures, num_returns=1) # only return one task
    ray.get(done)  # Get the result of the completed task