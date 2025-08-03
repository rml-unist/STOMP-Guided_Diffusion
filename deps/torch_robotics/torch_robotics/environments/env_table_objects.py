from copy import copy

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.robots import RobotPanda
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes

from torch_robotics.environments.objects import create_table_object_field, create_wall_field, create_cup_object, create_bottle_object, create_book_object

class EnvTableObject(EnvBase):
    def __init__(self, tensor_args=None, **kwargs):
        # table
        table_obj_field = create_table_object_field(tensor_args=tensor_args)
        table_sizes = table_obj_field.fields[0].sizes[0]
        dist_robot_to_table = 0.10
        theta = np.deg2rad(90)
        table_obj_field.set_position_orientation(
            pos=(dist_robot_to_table + table_sizes[1].item()/2, 0, -table_sizes[2].item()/2),
            ori=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
        )

        # wall behind
        limits = [[-1, -1, -1], [1.5, 1., 1.5]]
        wall_obs_field = create_wall_field(-0.2, limits, tensor_args=tensor_args)
        obj_list = [table_obj_field, wall_obs_field]

        # shelf object field
        cups = {1 : {'center' : (0.2, 0.0, 0.06), 'pos' : None, 'ori' : None},
                2 : {'center' : (0.5, 0.28, 0.11), 'pos' : None, 'ori' : None}}
        bottles = {1 : {'center' : (0.175, 0.06, 0.075), 'pos' : None, 'ori' : None},
                   2 : {'center' : (0.2, -0.25, 0.075), 'pos' : None, 'ori' : None}}
        book = {1 : {'center' : (0.52, 0.30, 0.025), 'pos' : None, 'ori' : None},
                2 : {'center' : (0.37, 0.0, 0.025), 'pos' : None, 'ori' : [np.cos(np.deg2rad(30) / 2), 0, 0, np.sin(np.deg2rad(30)/ 2)]},
                3 : {'center' : (0.37, 0.0, 0.075), 'pos' : None, 'ori' : [np.cos(np.deg2rad(35) / 2), 0, 0, np.sin(np.deg2rad(35)/ 2)]}}
        
        for key, value in cups.items():
            obj_field = create_cup_object(value['center'], key, tensor_args=tensor_args)
            obj_field.set_position_orientation(pos = value['pos'], ori=value['ori'])
            obj_list.append(obj_field)
        
        for key, value in bottles.items():
            obj_field = create_bottle_object(value['center'], key, tensor_args=tensor_args)
            obj_field.set_position_orientation(pos = value['pos'], ori=value['ori'])
            obj_list.append(obj_field)
        
        for key, value in book.items():
            obj_field = create_book_object(value['center'], key, tensor_args=tensor_args)
            obj_field.set_position_orientation(pos = value['pos'], ori=value['ori'])
            obj_list.append(obj_field)

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