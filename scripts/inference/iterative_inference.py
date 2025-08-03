from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

import os
import pickle
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
from mpd.models import TemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity, GuideManagerTrajectoriesWithSTOMP, \
                                                GuideManagerTrajectoriesWithAdversary
from mpd.models.diffusion_models.sample_functions import mean_ddpm_sample_fn, noise_ddpm_sample_fn, mean_guide_gradient_steps
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.robots import RobotPanda
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

from torch.distributions import MultivariateNormal

allow_ops_in_compiled_graph()

# Todo : make visualization part indepedent from this file using pkl files

TRAINED_MODELS_DIR = '/nas_data/MPD/data_trained_models/'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='EnvSpheres3D-RobotPanda')
parser.add_argument('--sample_ver', type=str, default='noise_ddpm')
parser.add_argument('--n_samples', type=int, default=50)
parser.add_argument('--guidance_samples', type=int, default=20)
parser.add_argument('--step_size', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=30)
parser.add_argument('--save_dir', type=str, default='./')

# @single_experiment_yaml
def experiment(
    ########################################################################################################################
    # Experiment configuration
    # model_id: str = 'EnvDense2D-RobotPointMass',
    # model_id: str = 'EnvNarrowPassageDense2D-RobotPointMass',
    # model_id: str = 'EnvSimple2D-RobotPointMass',
    model_id: str = 'EnvSpheres3D-RobotPanda',
    # planner_alg: str = 'diffusion_prior',
    # planner_alg: str = 'diffusion_prior_then_guide',
    sample_ver: str = 'noise_ddpm',
    chain_x0: bool = True, # visualize recon / latent

    guidance_sample: int = 10,
    n_samples: int = 50,
    n_support_points: int = 20,
    n_diffusion_steps_without_noise: int = 5,

    ########################################################################
    # MANDATORY
    seed: int = 30,
    results_dir: str = 'logs',
    save_dir: str = './',

    sample_fn_kwargs = None,
    task = None,
    dataset = None,
    model = None,
    ########################################################################
    **kwargs
):
    ########################################################################################################################
    
    fix_random_seed(seed)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!: ", seed, guidance_sample)

    # str_step_size = f'{step_size:.2g}'
    # results_dir = os.path.join(save_dir, model_id, sample_ver, str(seed), str_step_size)
    results_dir = os.path.join(save_dir, model_id, str(seed), str(guidance_sample))
    os.makedirs(results_dir, exist_ok=True)

    # Random initial and final positions
    n_tries = 100
    start_state_pos, goal_state_pos = None, None
    for _ in range(n_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state_pos = q_free[0]
        goal_state_pos = q_free[1]

        if torch.linalg.norm(start_state_pos - goal_state_pos) > dataset.threshold_start_goal_pos:
            break

    if start_state_pos is None or goal_state_pos is None:
        raise ValueError(f"No collision free configuration was found\n"
                         f"start_state_pos: {start_state_pos}\n"
                         f"goal_state_pos:  {goal_state_pos}\n")

    print(f'start_state_pos: {start_state_pos}')
    print(f'goal_state_pos: {goal_state_pos}')

    ########################################################################################################################
    # Run motion planning inference

    ########
    # normalize start and goal positions
    hard_conds = dataset.get_hard_conditions(torch.vstack((start_state_pos, goal_state_pos)), normalize=True)
    context = None

    ########
    # Sample trajectories with the diffusion/cvae model
    with TimerCUDA() as timer_model_sampling:
        trajs_normalized_iters = model.run_inference(
            context, hard_conds,
            n_samples=n_samples, horizon=n_support_points,
            return_chain=True,
            x0=chain_x0,
            sample_ver=sample_ver, # change it with a conditional
            **sample_fn_kwargs,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            # ddim=True
        )
    print(f't_model_sampling: {timer_model_sampling.elapsed:.3f} sec')
    t_total = timer_model_sampling.elapsed

    # unnormalize trajectory samples from the models
    trajs_iters = dataset.unnormalize_trajectories(trajs_normalized_iters)

    trajs_final = trajs_iters[-1]
    trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = task.get_trajs_collision_and_free(trajs_final, return_indices=True)

    ########################################################################################################################
    # Compute motion planning metrics
    print(f'\n----------------METRICS----------------')
    print(f't_total: {t_total:.3f} sec')

    success_free_trajs = task.compute_success_free_trajs(trajs_final)
    fraction_free_trajs = task.compute_fraction_free_trajs(trajs_final)
    collision_intensity_trajs = task.compute_collision_intensity_trajs(trajs_final)

    print(f'success: {success_free_trajs}')
    print(f'percentage free trajs: {fraction_free_trajs*100:.2f}')
    print(f'percentage collision intensity: {collision_intensity_trajs*100:.2f}')

    # compute costs only on collision-free trajectories
    traj_final_free_best = None
    idx_best_traj = None
    cost_best_free_traj = None
    cost_smoothness = None
    cost_path_length = None
    cost_all = None
    variance_waypoint_trajs_final_free = None
    if trajs_final_free is not None:
        cost_smoothness = compute_smoothness(trajs_final_free, robot)
        print(f'cost smoothness: {cost_smoothness.mean():.4f}, {cost_smoothness.std():.4f}')

        cost_path_length = compute_path_length(trajs_final_free, robot)
        print(f'cost path length: {cost_path_length.mean():.4f}, {cost_path_length.std():.4f}')

        # compute best trajectory
        cost_all = cost_path_length + cost_smoothness
        idx_best_traj = torch.argmin(cost_all).item()
        traj_final_free_best = trajs_final_free[idx_best_traj]
        cost_best_free_traj = torch.min(cost_all).item()
        print(f'cost best: {cost_best_free_traj:.3f}')

        # variance of waypoints
        variance_waypoint_trajs_final_free = compute_variance_waypoints(trajs_final_free, robot)
        print(f'variance waypoint: {variance_waypoint_trajs_final_free:.4f}')

    print(f'\n--------------------------------------\n')

    ########################################################################################################################
    # Save data
    results_data_dict = {
        # 'trajs_iters': trajs_iters,
        # 'trajs_final_coll': trajs_final_coll,
        # 'trajs_final_coll_idxs': trajs_final_coll_idxs,
        # 'trajs_final_free': trajs_final_free,
        # 'trajs_final_free_idxs': trajs_final_free_idxs,
        'success_free_trajs': success_free_trajs,
        # 'fraction_free_trajs': fraction_free_trajs,
        'collision_intensity_trajs': collision_intensity_trajs.item(),
        # 'idx_best_traj': idx_best_traj,
        # 'traj_final_free_best': traj_final_free_best,
        # 'cost_best_free_traj': cost_best_free_traj,
        'cost_path_length_trajs_final_free': cost_smoothness.min().item() if cost_smoothness is not None else None,
        'cost_smoothness_trajs_final_free': cost_path_length.min().item() if cost_path_length is not None else None,
        # 'cost_all_trajs_final_free': cost_all,
        'variance_waypoint_trajs_final_free': variance_waypoint_trajs_final_free.item() if variance_waypoint_trajs_final_free is not None else None,
        't_total': t_total
    }
    with open(os.path.join(results_dir, 'results_data_dict.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    save_dir = '/data/MPDresultsSTOMPSamplesExplore/'

    model_ids = [
        'EnvSpheres3D-RobotPanda',
        'EnvSimple2D-RobotPointMass',
        'EnvNarrowPassageDense2D-RobotPointMass',
        'EnvDense2D-RobotPointMass',
    ]

    sample_vers = ['noise_ddpm']

    seeds = list(range(0, 300))
    n_samples = 50

    # step_sizes = [3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005]

    step_size = 0.5
    guidance_sample = [5, 10, 15, 20, 25, 30, 40, 50]

    trajectory_duration = 5.0  # currently fixed
    planner_alg = 'mpd'
    render = False

    use_guide_on_extra_objects_only = False

    latent_guide = False #False: Universal guidance
    gradient_free_guide = True # True : STOMP
    gradient_free_guide_ver = 'STOMP' # 'STOMP', 'Adv'
    resample = False
    # guidance_sample = 10,
    start_guide_steps_fraction = 1
    n_guide_steps = 1
    n_diffusion_steps_without_noise = 5

    weight_grad_cost_collision = 1e-2 # 1
    weight_grad_cost_smoothness = 1e-7 # 1e-5
    noise_size = 1

    factor_num_interpolated_points_for_collision = 1.5

    device ='cuda:0'
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    for model_id in model_ids:
        # Define model
        print(f'##########################################################################################################')
        print(f'Model -- {model_id}')
        print(f'Algorithm -- {planner_alg}')
        run_prior_only = False
        run_prior_then_guidance = False
        if planner_alg == 'mpd':
            pass
        elif planner_alg == 'diffusion_prior_then_guide':
            run_prior_then_guidance = True
        elif planner_alg == 'diffusion_prior':
            run_prior_only = True
        else:
            raise NotImplementedError
        
        model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)

        args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

        # Load dataset with env, robot, task
        obstacle_cutoff_margin = 0.05
        if model_id == 'EnvSpheres3D-RobotPanda':
            # due to volume of the arm
            obstacle_cutoff_margin = 0.06

        train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        use_extra_objects=True,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        tensor_args=tensor_args,
        **args,
        )

        dataset = train_subset.dataset
        n_support_points = dataset.n_support_points
        env = dataset.env    
        robot = dataset.robot
        task = dataset.task

        dt = trajectory_duration / n_support_points  # time interval for finite differences

        # set robot's dt
        robot.dt = dt

        # Load prior model
        diffusion_configs = dict(
            variance_schedule=args['variance_schedule'],
            n_diffusion_steps=args['n_diffusion_steps'],
            predict_epsilon=args['predict_epsilon'],
        )
        unet_configs = dict(
            state_dim=dataset.state_dim,
            n_support_points=dataset.n_support_points,
            unet_input_dim=args['unet_input_dim'],
            dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
        )
        diffusion_model = get_model(
            model_class=args['diffusion_model_class'],
            model=TemporalUnet(**unet_configs),
            tensor_args=tensor_args,
            **diffusion_configs,
            **unet_configs
        )
        diffusion_model.load_state_dict(
            torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'),
            # torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_epoch_4729_iter_350000_state_dict.pth' if args['use_ema'] else 'model_epoch_4729_iter_350000state_dict.pth'),
            map_location=tensor_args['device'])
        )
        diffusion_model.eval()
        model = diffusion_model

        freeze_torch_model_params(model)
        model = torch.compile(model)
        model.warmup(horizon=n_support_points, device=device)

            ########
        # Set up the planning costs

        # Cost collisions
        # Cost collisions
        cost_collision_l = []
        weights_grad_cost_l = []  # for guidance, the weights_cost_l are the gradient multipliers (after gradient clipping)
        if use_guide_on_extra_objects_only:
            collision_fields = task.get_collision_fields_extra_objects()
        else:
            collision_fields = task.get_collision_fields()

        for collision_field in collision_fields:
            cost_collision_l.append(
                CostCollision(
                    robot, n_support_points,
                    field=collision_field,
                    sigma_coll=1.0,
                    tensor_args=tensor_args
                )
            )
            weights_grad_cost_l.append(weight_grad_cost_collision)

        # Cost smoothness
        cost_smoothness_l = [
            CostGPTrajectory(
                robot, n_support_points, dt, sigma_gp=1.0,
                tensor_args=tensor_args
            )
        ]
        weights_grad_cost_l.append(weight_grad_cost_smoothness)

        ####### Cost composition
        cost_func_list = [
            *cost_collision_l,
            *cost_smoothness_l
        ]

        cost_composite = CostComposite(
            robot, n_support_points, cost_func_list,
            weights_cost_l=weights_grad_cost_l,
            tensor_args=tensor_args
        )

        ########
        # Guiding manager
        sampler = None
        if gradient_free_guide:
            sampler = torch.distributions.MultivariateNormal(torch.zeros(2*robot.q_dim).to(**tensor_args), torch.eye(2*robot.q_dim).to(**tensor_args))
            if gradient_free_guide_ver == 'STOMP':
                GuideClass = GuideManagerTrajectoriesWithSTOMP
            elif gradient_free_guide_ver == 'Adv':
                GuideClass = GuideManagerTrajectoriesWithAdversary 
            else:
                if gradient_free_guide_ver is None:
                    raise ValueError("gradient_free_guide is None but gradient_free_guide = True")
                else:
                    raise NotImplementedError
        else:
            GuideClass = GuideManagerTrajectoriesWithVelocity

        guide = GuideClass(
                dataset,
                cost_composite,
                clip_grad=False,
                interpolate_trajectories_for_collision=True,
                num_interpolated_points=ceil(n_support_points * factor_num_interpolated_points_for_collision),
                tensor_args=tensor_args,
        )

        t_start_guide = ceil(start_guide_steps_fraction * model.n_diffusion_steps)

        sample_fn_kwargs = dict(
                guide=None if run_prior_then_guidance or run_prior_only else guide,
                gradient_free_guide_ver=gradient_free_guide_ver if gradient_free_guide else None,
                sample_num=0,
                sampler=sampler,
                cost = cost_composite if resample else None, # Add conditional here
                n_guide_steps=n_guide_steps,
                t_start_guide=t_start_guide,
                latent_guide=latent_guide,
                noise_std_extra_schedule_fn=lambda x: noise_size, 
                step_size=step_size,
                tensor_args=tensor_args
            )
        
        # Main experiment loops
        for sample_num in guidance_sample:
            sample_fn_kwargs['sample_num'] = sample_num

            for sample_ver in sample_vers:
                for seed in seeds:
                    # Keep tracking what env and robot are involved in
                    print(model_id, robot.q_dim)

                    experiment(model_id=model_id,
                            sample_ver = sample_ver,
                            guidance_sample = sample_num,
                            seed = seed,

                            n_samples = n_samples,
                            n_support_points = n_support_points,
                            n_diffusion_steps_without_noise = n_diffusion_steps_without_noise,

                            sample_fn_kwargs = sample_fn_kwargs,
                            model = model,
                            dataset = dataset,
                            task = task
                            )




