from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

import os
import pickle
import time

import torch
import yaml
from matplotlib import pyplot as plt
from pathlib import Path

from mpd.utils.loading import load_params_from_yaml
from mpd.trainer import get_dataset

# from experiment_launcher import single_experiment_yaml, run_experiment
from experiment_launcher.utils import fix_random_seed

from mp_baselines.planners.hybrid_planner import HybridPlanner
from mp_baselines.planners.multi_sample_based_planner import MultiSampleBasedPlanner
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.gpmp2 import GPMP2
from mp_baselines.planners.stoch_gpmp import StochGPMP

from torch_robotics.robots import RobotPanda
from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints

TRAINED_MODELS_DIR = '/nas_data/MPD/data_trained_models/'

def generate_trajectories_with_RRTC(
    env, robot, task,
    start_state_pos, goal_state_pos,
    num_trajectories,
    rrt_max_time=15,
    tensor_args=None,
    debug=False,
    **kwargs
):

    # Sample-based planner
    rrt_connect_default_params_env = env.get_rrt_connect_params(robot=robot)
    rrt_connect_default_params_env['max_time'] = rrt_max_time

    rrt_connect_params = dict(
        **rrt_connect_default_params_env,
        task=task,
        start_state_pos=start_state_pos,
        goal_state_pos=goal_state_pos,
        tensor_args=tensor_args,
    )
    planner = RRTConnect(**rrt_connect_params)
    # planner = MultiSampleBasedPlanner(
    #     sample_based_planner_base,
    #     n_trajectories=num_trajectories,
    #     max_processes=-1,
    #     optimize_sequentially=True
    # )

    # Optimize
    trajs_iters, t_total = planner.optimize(debug=debug, print_times=True, return_iterations=True, return_stat=True)
    return trajs_iters, t_total

def generate_trajectories_with_GPMP(
    env, robot, task,
    start_state_pos, goal_state_pos,
    num_trajectories,
    gpmp_opt_iters=30,
    n_support_points=64,
    duration=5.0,
    tensor_args=None,
    debug=False,
    **kwargs
):

    # Optimization-based planner
    dt = duration / n_support_points
    gpmp_default_params_env = env.get_gpmp2_params(robot=robot)
    gpmp_default_params_env['opt_iters'] = gpmp_opt_iters
    gpmp_default_params_env['n_support_points'] = n_support_points
    gpmp_default_params_env['dt'] = dt

    planner_params = dict(
        **gpmp_default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        num_particles_per_goal=num_trajectories,
        start_state=start_state_pos,
        multi_goal_states=goal_state_pos.unsqueeze(0),  # add batch dim for interface,
        collision_fields=task.get_collision_fields(),
        weights_cost_list = [1.0, 1.0, 1.0, 1.0, 1.0],
        tensor_args=tensor_args,
    )
    planner = GPMP2(**planner_params)

    # Optimize
    trajs_iters, t_total = planner.optimize(debug=debug, print_times=True, return_iterations=True, return_stat=True)
    return trajs_iters, t_total

def generate_trajectories_with_stochGPMP(
    env, robot, task,
    start_state_pos, goal_state_pos,
    num_trajectories,
    gpmp_opt_iters=60,
    n_support_points=64,
    duration=5.0,
    tensor_args=None,
    debug=False,
    **kwargs
):

    # Optimization-based planner : same parameters with gpmp2
    gpmp_default_params_env = env.get_gpmp2_params(robot=robot)
    gpmp_default_params_env['opt_iters'] = gpmp_opt_iters
    gpmp_default_params_env['n_support_points'] = n_support_points
    gpmp_default_params_env['dt'] = duration / n_support_points
    gpmp_default_params_env['num_samples'] = 2
    gpmp_default_params_env['step_size'] = 1

    planner_params = dict(
        **gpmp_default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        num_particles_per_goal=num_trajectories,
        start_state=start_state_pos,
        weights_cost_list = [1.0, 1.0, 1.0, 1.0, 1.0],
        multi_goal_states=goal_state_pos.unsqueeze(0),  # add batch dim for interface,
        initial_particle_means=None,
        collision_fields=task.get_collision_fields(),
        temperature=10,
        tensor_args=tensor_args,
    )
    planner = StochGPMP(**planner_params)

    # Optimize
    trajs_iters, t_total = planner.optimize(debug=debug, print_times=True, return_stat=True)
    return trajs_iters, t_total

def experiment(
    # model_id: str = 'EnvDense2D-RobotPointMass',
    # model_id: str = 'EnvNarrowPassageDense2D-RobotPointMass',
    # model_id: str = 'EnvSimple2D-RobotPointMass',
    model_id: str = 'EnvSpheres3D-RobotPanda',

    planner_alg: str = 'stochGPMP', #'RRTC', 'GPMP', 'stochGPMP'

    trajectory_duration: float = 5.0,  # seconds

    num_trajectories: int = 100,

    # device: str = 'cpu',
    device: str = 'cuda',
    debug: bool = True,
    render: bool = True,

    #######################################
    # MANDATORY
    seed: int = 50,

    results_dir: str = 'logs',
    save_dir: str = './results_inference',
    #######################################
    **kwargs
):
    fix_random_seed(seed)

    if planner_alg == 'RRTC':
        num_trajectories = 1 
    print(f'\n\n-------------------- Generating data --------------------')
    print(f'Seed:  {seed}')
    print(f'Alg: {planner_alg}')
    print(f'Env-Robot: {model_id}')  
    print(f'num_trajectories: {num_trajectories}')

    ####################################################################################################################
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    results_dir = os.path.join(save_dir, model_id, planner_alg, str(seed))
    os.makedirs(results_dir, exist_ok=True)

    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    # Load dataset with env, robot, task
    obstacle_cutoff_margin = 0.05
    if model_id == 'EnvSpheres3D-RobotPanda' and planner_alg != 'RRTC':
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

    dt = trajectory_duration / n_support_points  # time interval for finite differences
    robot.dt = dt # set robot's dt

    # -------------------------------- Start, Goal states ---------------------------------
    n_tries=100
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
    
    # Generate trajectories
    if planner_alg == 'RRTC':
        trajs_final, t_total = generate_trajectories_with_RRTC(
            env, robot, task,
            start_state_pos, goal_state_pos,
            num_trajectories,
            tensor_args=tensor_args,
            debug=debug,
        )
    elif planner_alg == 'GPMP':
        trajs_final, t_total = generate_trajectories_with_GPMP(
            env, robot, task,
            start_state_pos, goal_state_pos,
            num_trajectories,
            n_support_points=n_support_points,
            duration=trajectory_duration,
            tensor_args=tensor_args,
            debug=debug,
        )

    elif planner_alg == 'stochGPMP':
        trajs_final, t_total = generate_trajectories_with_stochGPMP(
            env, robot, task,
            start_state_pos, goal_state_pos,
            num_trajectories,
            n_support_points=n_support_points,
            duration=trajectory_duration,
            tensor_args=tensor_args,
            debug=debug,
        )

    # Compute motion planning metricss
    print(f'\n----------------METRICS----------------')
    print(f't_total: {t_total:.3f} sec')

    success_free_trajs=False
    fraction_free_trajs=0

    traj_final_free_best = None
    trajs_final_free = None
    idx_best_traj = None
    cost_best_free_traj = None
    cost_smoothness = None
    cost_path_length = None
    cost_all = None

    if trajs_final is not None:
        if planner_alg == 'RRTC':
            trajs_final = trajs_final.unsqueeze(0)
        trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = task.get_trajs_collision_and_free(trajs_final, return_indices=True)
        
        success_free_trajs = task.compute_success_free_trajs(trajs_final)
        fraction_free_trajs = task.compute_fraction_free_trajs(trajs_final)

        print(f'success: {success_free_trajs}')
        print(f'percentage free trajs: {fraction_free_trajs*100:.2f}')

        if planner_alg != 'RRTC':
            collision_intensity_trajs = task.compute_collision_intensity_trajs(trajs_final)
            print(f'percentage collision intensity: {collision_intensity_trajs*100:.2f}')

        variance_waypoint_trajs_final_free = None
        if trajs_final_free is not None:
            trajs_final_free_vel = None
            if planner_alg == 'RRTC':
                dt = trajectory_duration / trajs_final_free.shape[1]
                trajs_final_free_vel = (trajs_final_free[..., 1:] - trajs_final_free[..., :-1]) / dt
            cost_smoothness = compute_smoothness(trajs_final_free, robot, trajs_vel=trajs_final_free_vel)
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
            if planner_alg != 'RRTC':
                variance_waypoint_trajs_final_free = compute_variance_waypoints(trajs_final_free, robot)
                print(f'variance waypoint: {variance_waypoint_trajs_final_free:.4f}')

    else:
        print("No successful trajectory")
    print(f'\n--------------------------------------\n')

    ########################################################################################################################
    # Save data
    results_data_dict = {
        'success_free_trajs': success_free_trajs,
        'cost_path_length_trajs_final_free': cost_path_length.min().item() if cost_path_length is not None else None,
        'cost_smoothness_trajs_final_free': cost_smoothness.min().item() if cost_smoothness is not None else None,
        't_total': t_total
    }

    if planner_alg != 'RRTC':
        results_data_dict.update({'collision_intensity_trajs':collision_intensity_trajs.item()})
        results_data_dict.update({'variance_waypoint_trajs_final_free':variance_waypoint_trajs_final_free.item() if variance_waypoint_trajs_final_free is not None else None})
    
    with open(os.path.join(results_dir, 'results_data_dict.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ########################################################################################################################
    # Render results
    if render and trajs_final_free is not None:
        # Render
        planner_visualizer = PlanningVisualizer(
            task=task,
        )

        base_file_name = Path(os.path.basename(__file__)).stem

        # pos_trajs_iters = robot.get_position(trajs_iters)

        # planner_visualizer.animate_opt_iters_joint_space_state(
        #     trajs=trajs_iters,
        #     pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
        #     vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
        #     traj_best=traj_final_free_best,
        #     video_filepath=os.path.join(results_dir, f'{base_file_name}-joint-space-opt-iters.gif'),
        #     n_frames=max((2, len(trajs_iters))),
        #     anim_time=5
        # )
        # plt.close('all')

        if isinstance(robot, RobotPanda):
            # visualize in Isaac Gym
            # POSITION CONTROL
            # add initial positions for better visualization
            n_first_steps = 10
            n_last_steps = 10
            
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
                                    start_states_joint_pos=trajs_pos[0], goal_state_joint_pos=trajs_pos[-1][0],
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
            pass
            # visualize in the planning environment
            # trajs should be 4 dim (diffusion step + extra negative steps, batch, horizion, q_dim)
            # Make an animation: how sampled trajectories are changed through diffusion steps. 
            # planner_visualizer.animate_opt_iters_robots(
            #     trajs=pos_trajs_iters, start_state=start_state_pos, goal_state=goal_state_pos,
            #     traj_best=traj_final_free_best,
            #     video_filepath=os.path.join(results_dir, f'{base_file_name}-traj-opt-iters.gif'),
            #     n_frames=max((2, len(trajs_iters))),
            #     anim_time=5
            # )

            # planner_visualizer.plot_opt_iters_robot(
            #     trajs=pos_trajs_iters, start_state=start_state_pos, goal_state=goal_state_pos,
            #     draw_steps=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
            #     plot_filepath=os.path.join(results_dir, f'{base_file_name}-traj-opt-iters.png')
            # )

            # # trajs should be 3 dim (batch, horizion, q_dim)
            # # Make an animation: how a robot follow the trajectory planned at the final diffusion step
            # planner_visualizer.animate_robot_trajectories(
            #     trajs=pos_trajs_iters[-1], start_state=start_state_pos, goal_state=goal_state_pos,
            #     plot_trajs=True,
            #     video_filepath=os.path.join(results_dir, f'{base_file_name}-robot-traj.gif'),
            #     # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
            #     n_frames=pos_trajs_iters[-1].shape[1],
            #     anim_time=trajectory_duration
            # )
    return success_free_trajs

if __name__ == '__main__':
    success = 0
    for i in range(50):
        case_success = experiment(seed=i, render=False)
        success += case_success
        if (i+1) % 10 == 0:
            print("----------------------------------------------")
            print()
            print()
            print()
            print("success: ", success)
            print()
            print()
            print()
            print("----------------------------------------------")