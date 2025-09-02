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

def get_isotropic_sampler(
        dof,
        num_steps,
        tensor_args
):
    return torch.distributions.MultivariateNormal(torch.zeros(num_steps, 2*dof).to(**tensor_args),
                                                  torch.eye(2*dof).to(**tensor_args)) 

def get_STOMP_sampler(
        dt,
        dof,
        num_steps,
        sigma_start,
        sigma_goal,
        sigma_gp,
        tensor_args
):
    upper_diag = torch.diag(torch.ones(num_steps - 1), diagonal=1)
    lower_diag = torch.diag(torch.ones(num_steps - 1), diagonal=-1,)
    diag = -2 * torch.eye(num_steps)
    A_mat = upper_diag + diag + lower_diag
    A_mat = torch.cat(
        (torch.zeros(1, num_steps),
            A_mat,
            torch.zeros(1, num_steps)),
        dim=0,
    )
    A_mat[0, 0] = 1/sigma_start
    A_mat[-1, -1] = 1/sigma_goal
    A_mat = A_mat * 1./dt**2 * (1/sigma_gp)
    R_mat = A_mat.t() @ A_mat

    sampler = torch.distributions.MultivariateNormal(torch.zeros(2*dof ,num_steps).to(**tensor_args), precision_matrix=R_mat.to(**tensor_args))
    return sampler
    
def get_stochGPMP_sampler(
        dt,
        dof,
        num_steps,
        sigma_start,
        sigma_goal,
        sigma_gp,
        tensor_args,
        goal_directed=True
    ):
        # To construct the covariance we need to use float64, because the small covariances of the start, GP and goal
        # state distributions lead to very large inverse covariances during the construction of these matrices.
        # tensor_args['dtype'] = torch.float64
        state_dim = 2*dof
        M = 2 * dof * (num_steps)

        K_s_inv = (1/sigma_start)*torch.eye(2*dof, **tensor_args)
        K_g_inv = (1/sigma_goal)*torch.eye(2*dof, **tensor_args)
        K_gp_inv = (1/sigma_gp)*torch.concatenate([torch.concatenate([(12/dt**3)*torch.eye(dof), (-6/dt**2)*torch.eye(dof)]),
                                          torch.concatenate([(-6/dt**2)*torch.eye(dof), (4/dt)*torch.eye(dof)])], dim=1).to(**tensor_args)

        # Transition matrix
        Phi = torch.eye(state_dim, **tensor_args)
        Phi[:dof, dof:] = torch.eye(dof, **tensor_args) * dt
        diag_Phis = Phi
        for _ in range(num_steps - 2):
            diag_Phis = torch.block_diag(diag_Phis, Phi)

        A = torch.eye(M, **tensor_args)
        A[state_dim:, :-state_dim] += -1. * diag_Phis
        if goal_directed:
            b = torch.zeros(state_dim, M,  **tensor_args)
            b[:, -state_dim:] = torch.eye(state_dim,  **tensor_args)
            A = torch.cat((A, b))

        Q_inv = K_s_inv
        for _ in range(num_steps - 1):
            Q_inv = torch.block_diag(Q_inv, K_gp_inv).to(**tensor_args)
        if goal_directed:
            Q_inv = torch.block_diag(Q_inv, K_g_inv).to(**tensor_args)

        K_inv = (A.t() @ Q_inv @ A).to(**tensor_args) # M by M matrix
        sampler = torch.distributions.MultivariateNormal(torch.zeros(2*dof*num_steps).to(**tensor_args), precision_matrix=K_inv)
        return sampler

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
    planner_alg: str = 'mpd', #'mpd', 'diffusion_prior_then_stomp'
    sample_ver: str = 'noise_ddpm',
    chain_x0: bool = True, # visualize recon / latent

    use_guide_on_extra_objects_only: bool = False,

    n_samples: int = 100,

    # Todo : find the role of these variables
    latent_guide: bool = False, #False: Universal guidance
    gradient_free_guide: bool = True, # True : STOMP
    gradient_free_guide_ver: str = 'STOMP', # 'STOMP', 'Adv'

    resample: bool = False,
    guidance_sample: int = 15, # 2d: 5, panda: 15
    start_guide_steps_fraction: float = 1, 
    n_guide_steps: int = 1,
    n_diffusion_steps_without_noise: int = 5,
    n_post_gradient_step: int = 5,
    
    structured_noise: bool = False,
    recurrencing: bool = False,
    time_sample_ver: str = 'ddpm', # ddpm, ddim, ddpm_recurrencing, diffusion-es

    weight_grad_cost_collision: float = 1e-2, # 1e-2
    weight_grad_cost_smoothness: float = 1e-7, # 1e-7
    noise_size: float = 1,
    step_size: float = 5, 
    negative_step_size: float = 0.5, # 1

    factor_num_interpolated_points_for_collision: float = 1.5,

    trajectory_duration: float = 5.0,  # currently fixed

    ########################################################################
    device: str = 'cuda',

    debug: bool = True,

    render: bool = True,

    ########################################################################
    # MANDATORY
    seed: int = 30,
    results_dir: str = 'logs',
    save_dir: str = './',

    ########################################################################
    **kwargs
):
    ########################################################################################################################
    print(sample_ver, seed)
    fix_random_seed(seed)

    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    ########################################################################################################################
    print(f'##########################################################################################################')
    print(f'Model -- {model_id}')
    print(f'Algorithm -- {planner_alg}')
    run_prior_only = False
    run_prior_then_stomp = False
    if planner_alg == 'mpd':
        pass
    elif planner_alg == 'diffusion_prior_then_stomp':
        run_prior_then_stomp = True
    elif planner_alg == 'diffusion_prior':
        run_prior_only = True
    else:
        raise NotImplementedError
    ########################################################################################################################
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    str_step_size = f'{step_size:.2g}'

    # results_dir = os.path.join(save_dir, model_id, sample_ver, str(seed), str(guidance_sample))
    results_dir = os.path.join(save_dir, model_id, sample_ver, time_sample_ver, str(seed), str(guidance_sample))
    # results_dir = os.path.join(save_dir, model_id, sample_ver, str(n_post_gradient_step), str(seed), str_step_size)

    os.makedirs(results_dir, exist_ok=True)

    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    ########################################################################################################################
    # Load dataset with env, robot, task
    obstacle_cutoff_margin = 0.05
    if model_id == 'EnvSpheres3D-RobotPanda':
        # due to volume of the arm
        obstacle_cutoff_margin = 0.06

    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        use_extra_objects=True,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        **args,
        tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    env = dataset.env    
    robot = dataset.robot
    task = dataset.task

    # env = EnvTableShelf(tensor_args=tensor_args) #dataset.env   
    # env = EnvTableObject(tensor_args=tensor_args)
    # env = EnvShelfonTable(tensor_args=tensor_args)
    # robot = RobotPanda(grasped_object=GraspedObjectPandaBox(tensor_args=tensor_args), tensor_args=tensor_args) # dataset.robot
    # task = torch_robotics.tasks.tasks.PlanningTask(env=env, robot=robot, tensor_args=tensor_args) #dataset.task

    dt = trajectory_duration / n_support_points  # time interval for finite differences
    sigma_gp = 1.0

    # set robot's dt
    robot.dt = dt

    ########################################################################################################################
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

    ########################################################################################################################
     # Random initial and final positions
    n_tries = 100
    start_state_pos, goal_state_pos = None, None

    # table shelf
    # start_state_pos = torch.tensor([-0.8939,  0.9926,  0.9782, -0.9751, -0.9205,  0.7326, -0.9320]).to(**tensor_args)
    # goal_state_pos = torch.tensor([1.4238,  1.1257, -0.9062, -1.4260,  2.2422,  2.0772,  1.9869]).to(**tensor_args)

    # table shelf - grasped object
    # start_state_pos = torch.tensor([2.4578, -1.2870, -2.5946, -0.6008, -0.6913,  1.4121,  0.3714]).to(**tensor_args)
    # goal_state_pos = torch.tensor([ 0.5032,  0.6852,  0.1350, -1.3132,  1.5690,  1.7538, -1.2289]).to(**tensor_args)

    # table object - grapsed object
    # start_state_pos = torch.tensor([-0.0691,  0.3023, -0.6698, -2.2962,  0.2985,  2.5010, -1.7230]).to(**tensor_args)
    # goal_state_pos = torch.tensor([0.7186,  0.9205,  0.9156, -2.5072, -1.7072,  0.8746,  0.7100]).to(**tensor_args)

    # shelf on table - grasped object
    # Too cluttered : decreased collision margin in task.py line 255
    # start_state_pos = torch.tensor([0.4770,  0.6159, -0.4006, -2.4148, -1.4283,  1.7990, -2.5155]).to(**tensor_args)
    # goal_state_pos = torch.tensor([0.1287, -0.5467, -0.1503, -1.9593, -2.7611,  3.2674, -2.8131]).to(**tensor_args)

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
    # Set up the planning costs

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
            robot, n_support_points, dt, sigma_gp=sigma_gp,
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
    print("Creating a sampler...")
    sigma_start = 0.01
    sigma_goal = 0.01
    sigma_gp_sampler = 1
    dof = robot.q_dim

    # sampler = get_isotropic_sampler(dof, n_support_points, guidance_sample, n_samples, tensor_args)
    sampler = get_stochGPMP_sampler(dt, dof, n_support_points, 
                                    sigma_start, sigma_goal, sigma_gp_sampler, 
                                    tensor_args=tensor_args)
    # sampler = get_STOMP_sampler(dt, dof, n_support_points, 
    #                             sigma_start, sigma_goal, sigma_gp_sampler, tensor_args)

    if gradient_free_guide:
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

    if run_prior_then_stomp:
        gradient_estimator = GuideManagerTrajectoriesWithSTOMP(
                            dataset,
                            cost_composite,
                            clip_grad=False,
                            interpolate_trajectories_for_collision=True,
                            num_interpolated_points=ceil(n_support_points * factor_num_interpolated_points_for_collision),
                            tensor_args=tensor_args,
                            )
        
        
    t_start_guide = ceil(start_guide_steps_fraction * model.n_diffusion_steps)
        
    sample_fn_kwargs = dict(
        guide=None if run_prior_only else guide,
        gradient_free_guide_ver=gradient_free_guide_ver if gradient_free_guide else None,
        sample_num=guidance_sample,
        sampler=sampler,
        structured_noise=structured_noise,
        cost = cost_composite if resample else None,
        n_guide_steps=n_guide_steps,
        t_start_guide=t_start_guide,
        latent_guide=latent_guide,
        noise_std_extra_schedule_fn=lambda x: noise_size, 
        step_size=step_size,
        negative_step_size=negative_step_size,
        recurrencing=recurrencing,
        tensor_args=tensor_args
    )


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
            time_sample_ver=time_sample_ver
        )

    print(f't_model_sampling: {timer_model_sampling.elapsed:.3f} sec')
    t_total = timer_model_sampling.elapsed

    ########
    # run extra guiding steps without diffusion
    if run_prior_then_stomp:
        zero_tensor = torch.tensor([0]).to(device)      
        # print(SigmaInv.shape)                                                                                                                                                                                                                                                                                                                         

        with TimerCUDA() as timer_post_model_sample_guide:
            trajs = trajs_normalized_iters[-1].unsqueeze(0)
            _, batch, traj_len, state_dim = trajs.shape
            perturbe_shape = (guidance_sample, batch, traj_len, state_dim)
            
            trajs_post_diff_l = []
            for i in range(n_post_gradient_step):
                perturbation = sampler.sample(perturbe_shape[:2]).view(perturbe_shape)
                perturbed_trajs = trajs + 0.1*perturbation
                gradient = gradient_estimator.forward(perturbed_trajs, zero_tensor, perturbation, zero_tensor, guidance_sample, temperature=1e-4)
                # print(gradient.shape)

                trajs = trajs + 1e-2*gradient
                trajs_post_diff_l.append(trajs.squeeze(0))

            chain = torch.stack(trajs_post_diff_l, dim=1)
            chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')
            trajs_normalized_iters = torch.cat((trajs_normalized_iters, chain))
        print(f't_post_diffusion_guide: {timer_post_model_sample_guide.elapsed:.3f} sec')
        t_total = timer_model_sampling.elapsed + timer_post_model_sample_guide.elapsed


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
        'cost_path_length_trajs_final_free': cost_path_length.min().item() if cost_path_length is not None else None,
        'cost_smoothness_trajs_final_free': cost_smoothness.min().item() if cost_smoothness is not None else None,
        # 'cost_all_trajs_final_free': cost_all,
        'variance_waypoint_trajs_final_free': variance_waypoint_trajs_final_free.item() if variance_waypoint_trajs_final_free is not None else None,
        't_total': t_total
    }
    with open(os.path.join(results_dir, 'results_data_dict.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ########################################################################################################################
    # Render results
    if render:
        # Render
        planner_visualizer = PlanningVisualizer(
            task=task,
        )

        base_file_name = Path(os.path.basename(__file__)).stem

        pos_trajs_iters = robot.get_position(trajs_iters)

        planner_visualizer.animate_opt_iters_joint_space_state(
            trajs=trajs_iters,
            pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
            vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
            traj_best=traj_final_free_best,
            video_filepath=os.path.join(results_dir, f'{base_file_name}-joint-space-opt-iters.gif'),
            n_frames=max((2, len(trajs_iters))),
            anim_time=5
        )
        plt.close('all')

        if isinstance(robot, RobotPanda):
            # visualize in Isaac Gym
            # POSITION CONTROL
            # add initial positions for better visualization
            n_first_steps = 10
            n_last_steps = 10
            
            traj_num, _, _ = trajs_final_free.shape
            trajs_pos = robot.get_position(trajs_final_free).movedim(1, 0)          
            trajs_vel = robot.get_velocity(trajs_final_free).movedim(1, 0)

            trajs_pos = interpolate_traj_via_points(trajs_pos.movedim(0, 1), 2).movedim(1, 0)

            motion_planning_isaac_env = PandaMotionPlanningIsaacGymEnv(
                env, robot, task,
                asset_root="./deps/isaacgym/assets",
                controller_type='position',
                num_envs=trajs_pos.shape[1],
                all_robots_in_one_env=True,
                color_robots=False,
                color_robots_in_collision=True,
                show_goal_configuration=True,
                sync_with_real_time=True,
                dt=dt,
                **results_data_dict,
            )

            motion_planning_controller = MotionPlanningController(motion_planning_isaac_env)
            collision_free_idxs = motion_planning_controller.run_trajectories(
                                    trajs_pos,
                                    start_states_joint_pos=start_state_pos.view(1, -1).expand(traj_num, -1),
                                    goal_state_joint_pos=goal_state_pos,
                                    n_first_steps=n_first_steps,
                                    n_last_steps=n_last_steps,
                                    visualize=True,
                                    render_viewer_camera=True,
                                    make_video=True,
                                    video_path=os.path.join(results_dir, f'{base_file_name}-isaac-controller-position.mp4'),
                                    make_gif=True
                                )

            # Get the best trajectory among successful trajectories
            if len(collision_free_idxs) > 0:
                isaac_trajs_free = trajs_final_free[collision_free_idxs]
                isaac_cost_smoothness = compute_smoothness(isaac_trajs_free, robot)
                isaac_cost_path_length = compute_path_length(isaac_trajs_free, robot)
                isaac_cost_all = isaac_cost_path_length + isaac_cost_smoothness
                isaac_best_traj = isaac_trajs_free[torch.argmin(isaac_cost_all).item()]

                best_traj_pos = robot.get_position(isaac_best_traj).movedim(1, 0)
                best_traj_pos = interpolate_traj_via_points(best_traj_pos.movedim(0, 1), 2)

                sucess_data_dict = {
                    'isaac_free_trajs' : isaac_trajs_free,
                    'isaac_best_trajs' : isaac_best_traj
                }
               
                with open(os.path.join(results_dir, 'success_data_dict.pickle'), 'wb') as handle2:
                    pickle.dump(sucess_data_dict, handle2, protocol=pickle.HIGHEST_PROTOCOL)
                
                motion_planning_isaac_env = PandaMotionPlanningIsaacGymEnv(
                    env, robot, task,
                    asset_root="./deps/isaacgym/assets",
                    controller_type='position',
                    num_envs=1,
                    all_robots_in_one_env=True,
                    color_robots=False,
                    show_goal_configuration=True,
                    goal_as_start_vis=True, # use goal conf visualization for start conf visualization
                    sync_with_real_time=True,
                    dt=dt,
                    **results_data_dict,
                )
                motion_planning_controller = MotionPlanningController(motion_planning_isaac_env)
                motion_planning_controller.render_trajectory_positions(best_traj_pos,
                                                                    save_path=os.path.join(results_dir, f'{base_file_name}-isaac-trajectory.png'))

                fig, ax = planner_visualizer.render_robot_trajectories(traj_best=best_traj_pos,
                                                                    start_state=best_traj_pos[0],
                                                                    goal_state=best_traj_pos[-1])
                # For 75
                ax.view_init(elev=65, azim=60)
                ax.dist=7
                plt.savefig(os.path.join(results_dir, f'{base_file_name}-plt-trajectory.png'))

        else:
            # visualize in the planning environment
            # trajs should be 4 dim (diffusion step + extra negative steps, batch, horizion, q_dim)
            # Make an animation: how sampled trajectories are changed through diffusion steps. 
            planner_visualizer.animate_opt_iters_robots(
                trajs=pos_trajs_iters, start_state=start_state_pos, goal_state=goal_state_pos,
                traj_best=traj_final_free_best,
                video_filepath=os.path.join(results_dir, f'{base_file_name}-traj-opt-iters.gif'),
                n_frames=max((2, len(trajs_iters))),
                anim_time=5
            )

            planner_visualizer.plot_opt_iters_robot(
                trajs=pos_trajs_iters, start_state=start_state_pos, goal_state=goal_state_pos,
                draw_steps=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
                plot_filepath=os.path.join(results_dir, f'{base_file_name}-traj-opt-iters.png')
            )

            # trajs should be 3 dim (batch, horizion, q_dim)
            # Make an animation: how a robot follow the trajectory planned at the final diffusion step
            planner_visualizer.animate_robot_trajectories(
                trajs=pos_trajs_iters[-1], start_state=start_state_pos, goal_state=goal_state_pos,
                plot_trajs=True,
                video_filepath=os.path.join(results_dir, f'{base_file_name}-robot-traj.gif'),
                # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
                n_frames=pos_trajs_iters[-1].shape[1],
                anim_time=trajectory_duration
            )

        # plt.show()


if __name__ == '__main__':
    # Leave unchanged
    # run_experiment(experiment)
    # args = parser.parse_args()

    experiment(
        # model_id=args.model_id,
        # sample_ver=args.sample_ver,

        # n_samples=args.n_samples,
        # step_size=args.step_size,
        # guidance_sample=args.guidance_samples,
        
        # render=False,
        # seed=args.seed,
        # save_dir=args.save_dir,

        # debug=False,
    )