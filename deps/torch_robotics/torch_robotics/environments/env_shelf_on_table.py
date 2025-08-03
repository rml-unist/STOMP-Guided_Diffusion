from copy import copy

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.environments.objects import create_shelf_field, create_table_object_field, create_wall_field
from torch_robotics.robots import RobotPanda
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
    
class EnvShelfonTable(EnvBase):
    def __init__(self, tensor_args=None, **kwargs):
        # table object field
        table_obj_field = create_table_object_field(tensor_args=tensor_args)
        table_sizes = table_obj_field.fields[0].sizes[0]
        dist_robot_to_table = 0.10
        theta = np.deg2rad(90)
        table_obj_field.set_position_orientation(
            pos=(dist_robot_to_table + table_sizes[1].item()/2, 0, -table_sizes[2].item()/2),
            ori=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
        )

        # shelf object field
        shelf_obj_field = create_shelf_field(tensor_args=tensor_args, ontable=True)
        shelf_theta = np.deg2rad(-200)
        shelf_obj_field.set_position_orientation(
            pos = (0.6, -0.15, 0),
            ori=[np.cos(shelf_theta / 2), 0, 0, np.sin(shelf_theta / 2)]
        )
        
        limits = [[-1, -1, -1], [1.5, 1., 1.5]]
        wall_obs_field = create_wall_field(-0.5, limits, tensor_args=tensor_args)

        small_wall_limits = [[-0.5, -0.2, 0],[0.5, 0.2, 0.5]]
        wall_on_desk_field = create_wall_field(-0.25, small_wall_limits, tensor_args=tensor_args, wall_idx=1)
        wall_theta = np.deg2rad(-90)
        wall_on_desk_field.set_position_orientation(
            pos=(0.45, 0.0, 0),
            ori=[np.cos(wall_theta / 2), 0, 0, np.sin(wall_theta / 2)]
        )

        obj_list = [table_obj_field, shelf_obj_field, wall_obs_field] #, wall_on_desk_field]

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor(limits, **tensor_args),  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=250,
            num_samples=64,
            sigma_start=1e-3,
            sigma_gp=1e-1,
            sigma_goal_prior=1e-3,
            sigma_coll=1e-4,
            step_size=5e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.1,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )
        if isinstance(robot, RobotPanda):
            return params
        else:
            raise NotImplementedError

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=torch.pi/80,
            n_radius=torch.pi/4,
            n_pre_samples=50000,
            max_time=15
        )
        if isinstance(robot, RobotPanda):
            return params
        else:
            raise NotImplementedError
        