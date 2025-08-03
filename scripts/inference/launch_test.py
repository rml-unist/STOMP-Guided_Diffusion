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
# LAUNCHER

# LOCAL = is_local()
# TEST = False
# USE_CUDA = True

# N_SEEDS = 1

# N_EXPS_IN_PARALLEL = 1

# N_CORES = N_EXPS_IN_PARALLEL * 4
# MEMORY_SINGLE_JOB = 12000
# MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
# PARTITION = 'gpu_3090' if USE_CUDA else 'amd3,amd2,amd'
# GRES = 'gpu:rtx3090:3' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1, gpu:rtx3090:1, gpu:a5000:1
# CONDA_ENV = 'mpd'

# exp_name = f'test_diffusion'

# launcher = Launcher(
#     exp_name=exp_name,
#     exp_file='inference',
#     # project_name='project01234',
#     n_seeds=N_SEEDS,
#     n_exps_in_parallel=N_EXPS_IN_PARALLEL,
#     n_cores=N_CORES,
#     memory_per_core=MEMORY_PER_CORE,
#     days=5,
#     hours=23,
#     minutes=59,
#     seconds=0,
#     partition=PARTITION,
#     conda_env=CONDA_ENV,
#     gres=GRES,
#     use_timestamp=True
# )

ray.init(num_gpus=2, num_cpus=8)

@ray.remote(num_gpus=1, num_cpus=4)
def one_experiment(model_id, seed, sample_ver, n_samples, step_size, stomp_step, save_dir):
    sample_num=5 #15
    if model_id == 'EnvSpheres3D-RobotPanda':
        sample_num=15 # 25

    experiment(
        model_id=model_id,
        sample_ver=sample_ver,

        n_samples=n_samples,
        step_size=step_size,
        negative_step_size=step_size,
        n_post_gradient_step=stomp_step,

        guidance_sample=sample_num,
        
        render=False,
        seed=seed,
        save_dir=save_dir,

        debug=False,
    )

########################################################################################################################
# EXPERIMENT PARAMETERS SETUP
save_dir = '/data/MPDresultsVanillaSTOMPFixed3_Dense/'

model_ids = [
    # 'EnvSpheres3D-RobotPanda',
    # 'EnvSimple2D-RobotPointMass',
    # 'EnvNarrowPassageDense2D-RobotPointMass',
    'EnvDense2D-RobotPointMass',
]

sample_vers = ['noise_ddpm']

seeds = list(range(0, 300))
n_samples = 100

step_sizes = [3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005] #[25, 20, 15, 10]
stomp_num_steps = [5, 10, 15]

render = False

########################################################################################################################
# RUN
futures = []
for i, (model_id, seed, sample_ver, step_size, stomp_step) in enumerate(product(model_ids, seeds, sample_vers, step_sizes, stomp_num_steps)):
    # launcher.add_experiment(
    #     model_id__=model_id,
    #     sample_ver__=sample_ver,

    #     n_samples=n_samples,
    #     step_size=step_size,
    #     guidance_sample__=sample_num,
        
    #     render=False,
    #     seed__=seed,
    #     save_dir=save_dir,

    #     debug=False,
    # )
#     # Add .remote() here to submit each experiment as a Ray task
    future = one_experiment.remote(model_id, seed, sample_ver, n_samples, step_size, stomp_step, save_dir)
    futures.append(future)
#     # experiment(
#     #     model_id=model_id,
#     #     sample_ver=sample_ver,

#     #     n_samples=n_samples,
#     #     step_size=step_size,
        
#     #     render=False,
        
#     #     device=f"cuda:{gpu_ids[i % len(gpu_ids)]}",
#     #     debug=False,
#     # )

# launcher.run(LOCAL, TEST)
while futures:
    done, futures = ray.wait(futures, num_returns=1) # only return one task
    ray.get(done)  # Get the result of the completed task