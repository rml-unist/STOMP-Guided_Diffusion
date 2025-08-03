import abc

import torch
import numpy as np
from urdf_parser_py.urdf import Box

from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import to_numpy


class GraspedObject(ObjectField):

    def __init__(self, primitive_fields, **kwargs):
        # pos, ori - position and orientation are specified wrt to the end-effector link

        # Only one primitive type
        assert len(primitive_fields) == 1

        super().__init__(primitive_fields, **kwargs)

        # Geometry URDF
        self.geometry_urdf = self.get_geometry_urdf()

    def get_geometry_urdf(self):
        primitive_field = self.fields[0]

        if isinstance(primitive_field, MultiBoxField):
            size = primitive_field.sizes[0]
            return Box(to_numpy(size))
        else:
            raise NotImplementedError

    @abc.abstractmethod
    def get_base_points_for_collision(self):
        raise NotImplementedError


class GraspedObjectPandaBox(GraspedObject):

    def __init__(self, tensor_args=None, **kwargs):
        # One box
        primitive_fields = [
            MultiBoxField(torch.zeros(3, **tensor_args).view(1, -1),
                          torch.tensor([0.05, 0.05, 0.15], **tensor_args).view(1, -1),
                          tensor_args=tensor_args)
        ]

        # position and orientation wrt to the robots's end-effector link -> for panda reference_frame='panda_hand'

        # For table shelf
        pos = torch.tensor([0., 0., 0.11], **tensor_args)
        ori = torch.tensor([0, 0.7071081, 0, 0.7071055], **tensor_args)
        
        # For table objects
        # pos = torch.tensor([0., 0., 0.15], **tensor_args)
        # ori = torch.tensor([0.7071081, 0, 0, 0.7071055], **tensor_args)

        super().__init__(
            primitive_fields,
            name='GraspedObjectPandaBox',
            pos=pos, ori=ori, reference_frame='panda_hand',
            **kwargs)

        self.base_points_for_collision = self.get_base_points_for_collision()
        self.n_base_points_for_collision = len(self.base_points_for_collision)

    def get_base_points_for_collision(self):
        # points on vertices and centers of faces
        size = self.fields[0].sizes[0]
        x, y, z = size
        vertices = torch.tensor(
            [
                [x/2, y/2, -z/2],
                [x/2, -y/2, -z/2],
                [-x/2, -y/2, -z/2],
                [-x/2, y/2, -z/2],
                [x/2, y/2, z/2],
                [x/2, -y/2, z/2],
                [-x/2, -y/2, z/2],
                [-x/2, y/2, z/2],
            ],
            **self.tensor_args)

        faces = torch.tensor(
            [
                [x/2, 0, 0],
                [0, -y/2, 0],
                [-x/2, 0, 0],
                [0, y/2, 0],
                [0, 0, z/2],
                [0, 0, -z/2],
            ],
            **self.tensor_args)

        points = torch.cat((vertices, faces), dim=0)
        return points

def create_table_object_field(tensor_args=None):
    centers = [(0., 0., 0.)]
    sizes = [(0.9, 0.56, 0.80)] #[(0.56, 0.90, 0.80)]
    centers = np.array(centers)
    sizes = np.array(sizes)
    boxes = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    return ObjectField([boxes], 'table')


def create_shelf_field(tensor_args=None, ontable=False):
    width = 0.80 
    height = 2.05
    depth = 0.28
    side_panel_width = 0.02
    if ontable:
        width  *= 0.5
        height = 0.46

    shelf_width = width - 2*side_panel_width
    shelf_height = 0.015
    shelf_depth = depth

    # left panel
    centers = [(side_panel_width/2, depth/2, height/2)]
    sizes = [(side_panel_width, depth, height)]
    # right panel
    centers.append((side_panel_width + shelf_width + side_panel_width/2, depth/2, height/2))
    sizes.append((side_panel_width, depth, height))
    # back panel
    centers.append((side_panel_width + shelf_width/2, depth + side_panel_width/2, height/2))
    sizes.append((shelf_width, side_panel_width, height))

    # bottom shelf
    centers.append((side_panel_width + shelf_width/2, depth/2, shelf_height/2))
    sizes.append((shelf_width, shelf_depth, shelf_height))

    # top shelf
    centers.append((side_panel_width + shelf_width/2, depth/2, height - shelf_height/2))
    sizes.append((shelf_width, shelf_depth, shelf_height))

    # shelf 1
    first_shelf_height = 0.82 if not ontable else 0
    centers.append((side_panel_width + shelf_width/2, depth/2, first_shelf_height + shelf_height/2))
    sizes.append((shelf_width, shelf_depth, shelf_height))

    # next shelves
    plus_height_l = [height/2] if ontable else [0.23, 0.255, 0.225, 0.225]
    for plus_height in plus_height_l:
        center = list(copy(centers[-1]))
        center[-1] += plus_height
        centers.append(center)
        sizes.append((shelf_width, shelf_depth, shelf_height))

    centers = np.array(centers)
    # main_center = np.array((side_panel_width + shelf_width/2, depth/2, height/2))
    # centers -= main_center
    sizes = np.array(sizes)
    boxes = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    return ObjectField([boxes], 'shelf')

def create_wall_field(x_pos, limits, tensor_args=None, wall_idx=0):
    wall_width = 0.05
    center = np.array([[x_pos-wall_width/2, (limits[0][1] + limits[1][1])/2, (limits[0][2] + limits[1][2])/2]])
    size = np.array([[wall_width, limits[1][1] - limits[0][1], limits[1][2] - limits[0][2]]])
    box = MultiBoxField(center, size, tensor_args=tensor_args)
    return ObjectField([box], f'wall_{wall_idx}')

def create_cup_object(center, cup_id, tensor_args=None):
    # cups
    cup_dia = 0.07
    cup_height = 0.12

    centers = np.array([center])
    sizes = np.array([(cup_dia, cup_dia, cup_height)])
    
    box = MultiBoxField(centers=centers, sizes=sizes, tensor_args=tensor_args)
    return  ObjectField([box], f'cup_{cup_id}')

def create_bottle_object(center, bottle_id, tensor_args=None):
    bottle_dia = 0.05
    bottle_height = 0.15
    centers = np.array([center])
    sizes = np.array([(bottle_dia, bottle_dia, bottle_height)])

    box = MultiBoxField(centers=centers, sizes=sizes, tensor_args=tensor_args)
    return ObjectField([box], f'bottle_{bottle_id}')

def create_book_object(center, book_id, tensor_args=None):
    book_xwidth = 0.15
    book_ywidth = 0.18
    book_height = 0.05
    centers = np.array([center])
    sizes = np.array([(book_xwidth, book_ywidth, book_height)])

    box = MultiBoxField(centers=centers, sizes=sizes, tensor_args=tensor_args)
    return ObjectField([box], f'bottle_{book_id}')