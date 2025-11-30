from urchin import (
    URDF,
    URDFTypeWithMesh,
    Joint,
    Link,
    Transmission,
    Material,
    Inertial,
    Visual,
    Collision,
    JointLimit
)
from typing import Optional
from urchin.utils import parse_origin
from collections import OrderedDict
from lxml import etree as ET
import os
import torch
import numpy as np

from DoorOpening.utils.point_utils import tensor_to_ply

JointLimit._ATTRIBS['effort'] = (float, False)
JointLimit._ATTRIBS['velocity'] = (float, False)

def configure_origin(value, device=None):
    """Convert a value into a 4x4 transform matrix.
    Parameters
    ----------
    value : None, (6,) float, or (4,4) float
        The value to turn into the matrix.
        If (6,), interpreted as xyzrpy coordinates.
    Returns
    -------
    matrix : (4,4) float or None
        The created matrix.
    """
    assert isinstance(
        value, torch.Tensor
    ), "Invalid type for origin, expect 4x4 torch tensor"
    assert value.shape == (4, 4)
    return value.to(device)


class TorchVisual(Visual):
    def __init__(self, geometry, name=None, origin=None, material=None, device=None):
        self.device = device
        super().__init__(geometry, name, origin, material)

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value, self.device)

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        kwargs = cls._parse(node, path, lazy_load_meshes)
        kwargs["origin"] = torch.tensor(parse_origin(node))
        kwargs["device"] = device
        return TorchVisual(**kwargs)


class TorchCollision(Collision):
    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        kwargs = cls._parse(node, path, lazy_load_meshes)
        kwargs["origin"] = parse_origin(node)
        return TorchCollision(**kwargs)


class TorchLink(Link):
    _ELEMENTS = {
        "inertial": (Inertial, False, False),
        "visuals": (TorchVisual, False, True),
        "collisions": (TorchCollision, False, True),
    }

    def __init__(self, name, inertial, visuals, collisions, device=None):
        self.device = device
        super().__init__(name, inertial, visuals, collisions)

    @classmethod
    def _parse_simple_elements(cls, node, path, lazy_load_meshes, device):
        """Parse all elements in the _ELEMENTS array from the children of
        this node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse children for.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from element names to the :class:`URDFType` subclass (or list,
            if ``multiple`` was set) created for that element.
        """
        kwargs = {}
        for a in cls._ELEMENTS:
            t, r, m = cls._ELEMENTS[a]
            if not m:
                v = node.find(t._TAG)
                if r or v is not None:
                    if issubclass(t, URDFTypeWithMesh):
                        v = t._from_xml(v, path, lazy_load_meshes)
                    else:
                        v = t._from_xml(v, path)
            else:
                vs = node.findall(t._TAG)
                if len(vs) == 0 and r:
                    raise ValueError(
                        "Missing required subelement(s) of type {} when "
                        "parsing an object of type {}".format(t.__name__, cls.__name__)
                    )
                if issubclass(t, URDFTypeWithMesh):
                    v = [t._from_xml(n, path, lazy_load_meshes, device) for n in vs]
                else:
                    v = [t._from_xml(n, path, device) for n in vs]
            kwargs[a] = v
        return kwargs

    @classmethod
    def _parse(cls, node, path, lazy_load_meshes, device):
        """Parse all elements and attributes in the _ELEMENTS and _ATTRIBS
        arrays for a node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from names to Python classes created from the attributes
            and elements in the class arrays.
        """
        kwargs = cls._parse_simple_attribs(node)
        kwargs.update(cls._parse_simple_elements(node, path, lazy_load_meshes, device))
        return kwargs

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        """Create an instance of this class from an XML node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        obj : :class:`URDFType`
            An instance of this class parsed from the node.
        """
        return cls(**cls._parse(node, path, lazy_load_meshes, device))

class TorchLimit(JointLimit):
    def __init__(
        self,
        effort: Optional[float] = 1000.0,
        velocity: Optional[float] = 1.0,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ):
        if effort is None:
            effort = 1000.0
        if velocity is None:
            velocity = 1.0
        self.effort = effort
        self.velocity = velocity
        self.lower = lower
        self.upper = upper

class TorchJoint(Joint):
    _ELEMENTS = {
        "limit": (TorchLimit, False, False),
    }
    def __init__(
        self,
        name,
        joint_type,
        parent,
        child,
        axis=None,
        origin=None,
        limit=None,
        dynamics=None,
        safety_controller=None,
        calibration=None,
        mimic=None,
        device=None,
    ):
        self.device = device
        super().__init__(
            name,
            joint_type,
            parent,
            child,
            axis,
            origin,
            limit,
            dynamics,
            safety_controller,
            calibration,
            mimic,
        )

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value, device=self.device)

    @property
    def axis(self):
        """(3,) float : The joint axis in the joint frame."""
        return self._axis

    @axis.setter
    def axis(self, value):
        if value is None:
            value = torch.as_tensor([1.0, 0.0, 0.0], device=self.device)
        elif isinstance(value, torch.Tensor):
            assert value.shape == (3,), "Invalid shape for axis, should be (3,)"
            value = value.to(self.device)
            value = value / torch.norm(value)
        else:
            value = torch.as_tensor(value, device=self.device)
            if value.shape != (3,):
                raise ValueError("Invalid shape for axis, should be (3,)")
            value = value / torch.norm(value)
        self._axis = value

    @classmethod
    def _from_xml(cls, node, path, device):
        kwargs = cls._parse(node, path)
        kwargs["joint_type"] = str(node.attrib["type"])
        kwargs["parent"] = node.find("parent").attrib["link"]
        kwargs["child"] = node.find("child").attrib["link"]
        axis = node.find("axis")
        if axis is not None:
            axis = torch.as_tensor(
                np.fromstring(axis.attrib["xyz"], sep=" "),
            )
        kwargs["axis"] = axis
        kwargs["origin"] = torch.tensor(parse_origin(node))
        kwargs["device"] = device

        return TorchJoint(**kwargs)

    def _rotation_matrices(self, angles, axis):
        """Compute rotation matrices from angle/axis representations.
        Parameters
        ----------
        angles : (n,) float
            The angles.
        axis : (3,) float
            The axis.
        Returns
        -------
        rots : (n,4,4)
            The rotation matrices
        """
        axis = axis / torch.norm(axis)
        sina = torch.sin(angles)
        cosa = torch.cos(angles)
        M = torch.eye(4, device=self.device).repeat((len(angles), 1, 1))
        M[:, 0, 0] = cosa
        M[:, 1, 1] = cosa
        M[:, 2, 2] = cosa
        M[:, :3, :3] += (
            torch.ger(axis, axis).repeat((len(angles), 1, 1))
            * (1.0 - cosa)[:, np.newaxis, np.newaxis]
        )
        M[:, :3, :3] += (
            torch.tensor(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                device=self.device,
            ).repeat((len(angles), 1, 1))
            * sina[:, np.newaxis, np.newaxis]
        )
        return M

    def get_child_poses(self, cfg, n_cfgs):
        """Computes the child pose relative to a parent pose for a given set of
        configuration values.
        Parameters
        ----------
        cfg : (n,) float or None
            The configuration values for this joint. They are interpreted
            based on the joint type as follows:
            - ``fixed`` - not used.
            - ``prismatic`` - a translation along the axis in meters.
            - ``revolute`` - a rotation about the axis in radians.
            - ``continuous`` - a rotation about the axis in radians.
            - ``planar`` - Not implemented.
            - ``floating`` - Not implemented.
            If ``cfg`` is ``None``, then this just returns the joint pose.
        Returns
        -------
        poses : (n,4,4) float
            The poses of the child relative to the parent.
        """
        if cfg is None:
            return self.origin.repeat((n_cfgs, 1, 1))
        elif self.joint_type == "fixed":
            return self.origin.repeat((n_cfgs, 1, 1))
        elif self.joint_type in ["revolute", "continuous"]:
            if cfg is None:
                cfg = torch.zeros(n_cfgs)
            return torch.matmul(
                self.origin.type_as(cfg),
                self._rotation_matrices(cfg, self.axis).type_as(cfg),
            )
        elif self.joint_type == "prismatic":
            if cfg is None:
                cfg = torch.zeros(n_cfgs)
            translation = torch.eye(4, device=self.device).repeat((n_cfgs, 1, 1))
            translation[:, :3, 3] = self.axis * cfg[:, np.newaxis]
            return torch.matmul(self.origin.type_as(cfg), translation.type_as(cfg))
        elif self.joint_type == "planar":
            raise NotImplementedError()
        elif self.joint_type == "floating":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid configuration")


class TorchURDF(URDF):

    _ELEMENTS = {
        "links": (TorchLink, True, True),
        "joints": (TorchJoint, False, True),
        "transmissions": (Transmission, False, True),
        "materials": (Material, False, True),
    }

    def __init__(
        self,
        name,
        links,
        joints=None,
        transmissions=None,
        materials=None,
        other_xml=None,
        device=None,
    ):
        self.device = device
        super().__init__(name, links, joints, transmissions, materials, other_xml)

    @staticmethod
    def load(file_obj, lazy_load_meshes=True, device=None):
        """Load a URDF from a file.
        Parameters
        ----------
        file_obj : str or file-like object
            The file to load the URDF from. Should be the path to the
            ``.urdf`` XML file. Any paths in the URDF should be specified
            as relative paths to the ``.urdf`` file instead of as ROS
            resources.
        Returns
        -------
        urdf : :class:`.URDF`
            The parsed URDF.
        """
        if isinstance(file_obj, str):
            if os.path.isfile(file_obj):
                parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
                tree = ET.parse(file_obj, parser=parser)
                path, _ = os.path.split(file_obj)
            else:
                raise ValueError("{} is not a file".format(file_obj))
        else:
            parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
            tree = ET.parse(file_obj, parser=parser)
            path, _ = os.path.split(file_obj.name)

        node = tree.getroot()
        return TorchURDF._from_xml(node, path, lazy_load_meshes, device)

    # @classmethod
    # def _parse_simple_elements(cls, node, path, lazy_load_meshes, device):
    #     """Parse all elements in the _ELEMENTS array from the children of
    #     this node.
    #     Parameters
    #     ----------
    #     node : :class:`lxml.etree.Element`
    #         The node to parse children for.
    #     path : str
    #         The string path where the XML file is located (used for resolving
    #         the location of mesh or image files).
    #     Returns
    #     -------
    #     kwargs : dict
    #         Map from element names to the :class:`URDFType` subclass (or list,
    #         if ``multiple`` was set) created for that element.
    #     """
    #     kwargs = {}
    #     for a in cls._ELEMENTS:
    #         t, r, m = cls._ELEMENTS[a]
    #         if not m:
    #             v = node.find(t._TAG)
    #             if r or v is not None:
    #                 if issubclass(t, URDFTypeWithMesh):
    #                     v = t._from_xml(v, path, lazy_load_meshes)
    #                 else:
    #                     v = t._from_xml(v, path)
    #         else:
    #             vs = node.findall(t._TAG)
    #             if len(vs) == 0 and r:
    #                 raise ValueError(
    #                     "Missing required subelement(s) of type {} when "
    #                     "parsing an object of type {}".format(t.__name__, cls.__name__)
    #                 )
    #             if issubclass(t, URDFTypeWithMesh):
    #                 v = [t._from_xml(n, path, lazy_load_meshes, device) for n in vs]
    #             else:
    #                 print(t)
    #                 v = [t._from_xml(n, path, device) for n in vs]
    #         kwargs[a] = v
    #     return kwargs

    @classmethod
    def _parse_simple_elements(cls, node, path, lazy_load_meshes, device):
        """Parse all elements in the _ELEMENTS array from the children of
        this node.
        """
        kwargs = {}
        for a in cls._ELEMENTS:
            t, r, m = cls._ELEMENTS[a]

            if not m:
                v = node.find(t._TAG)
                if r or v is not None:
                    # Custom torch-aware types
                    if t is TorchLink:
                        v = t._from_xml(v, path, lazy_load_meshes, device)
                    elif t is TorchJoint:
                        v = t._from_xml(v, path, device)
                    # Standard urchin types with meshes
                    elif issubclass(t, URDFTypeWithMesh):
                        v = t._from_xml(v, path, lazy_load_meshes)
                    # Standard urchin types without meshes
                    else:
                        v = t._from_xml(v, path)
            else:
                vs = node.findall(t._TAG)
                if len(vs) == 0 and r:
                    raise ValueError(
                        "Missing required subelement(s) of type {} when "
                        "parsing an object of type {}".format(t.__name__, cls.__name__)
                    )

                if t is TorchLink:
                    v = [t._from_xml(n, path, lazy_load_meshes, device) for n in vs]
                elif t is TorchJoint:
                    v = [t._from_xml(n, path, device) for n in vs]
                elif issubclass(t, URDFTypeWithMesh):
                    v = [t._from_xml(n, path, lazy_load_meshes) for n in vs]
                else:
                    # Material, Transmission, etc. — no device arg
                    v = [t._from_xml(n, path) for n in vs]

            kwargs[a] = v

        return kwargs
    
    @classmethod
    def _parse(cls, node, path, lazy_load_meshes, device):
        """Parse all elements and attributes in the _ELEMENTS and _ATTRIBS
        arrays for a node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from names to Python classes created from the attributes
            and elements in the class arrays.
        """
        kwargs = cls._parse_simple_attribs(node)
        kwargs.update(cls._parse_simple_elements(node, path, lazy_load_meshes, device))
        return kwargs

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        valid_tags = set(["joint", "link", "transmission", "material"])
        kwargs = cls._parse(node, path, lazy_load_meshes, device)

        extra_xml_node = ET.Element("extra")
        for child in node:
            if child.tag not in valid_tags:
                extra_xml_node.append(child)

        data = ET.tostring(extra_xml_node)
        kwargs["other_xml"] = data
        kwargs["device"] = device
        return cls(**kwargs)

    def _process_cfgs(self, cfgs):
        """Process a list of joint configurations into a dictionary mapping joints to
        configuration values.
        This should result in a dict mapping each joint to a list of cfg values, one
        per joint.
        """
        joint_cfg = {}
        assert isinstance(cfgs, torch.Tensor), "Incorrectly formatted config array"
        n_cfgs = len(cfgs)
        for i, j in enumerate(self.actuated_joints):
            joint_cfg[j] = cfgs[:, i]

        return joint_cfg, n_cfgs

    def link_fk(self, cfg=None, link=None, links=None, use_names=False):
        raise NotImplementedError("Not implemented")

    def link_fk_batch(self, cfgs=None, use_names=False):
        """Computes the poses of the URDF's links via forward kinematics in a batch.
        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        use_names : bool
            If True, the returned dictionary will have keys that are string
            link names rather than the links themselves.
        Returns
        -------
        fk : dict or (n,4,4) float
            A map from links to a (n,4,4) vector of homogenous transform matrices that
            position the links relative to the base link's frame
        """
        joint_cfgs, n_cfgs = self._process_cfgs(cfgs)

        # Process link set
        link_set = self.links

        # Compute FK mapping each link to a vector of matrices, one matrix per cfg
        fk = OrderedDict()
        for lnk in self._reverse_topo:
            if lnk not in link_set:
                continue
            poses = torch.eye(4, device=self.device).repeat((n_cfgs, 1, 1))
            path = self._paths_to_base[lnk]
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                joint = self._G.get_edge_data(child, parent)["joint"]

                cfg_vals = None
                if joint.mimic is not None:
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfgs:
                        cfg_vals = joint_cfgs[mimic_joint]
                        cfg_vals = (
                            joint.mimic.multiplier * cfg_vals + joint.mimic.offset
                        )
                elif joint in joint_cfgs:
                    cfg_vals = joint_cfgs[joint]

                child_poses = joint.get_child_poses(cfg_vals, n_cfgs)
                poses = torch.matmul(child_poses, poses.type_as(child_poses))

                if parent in fk:
                    poses = torch.matmul(fk[parent], poses.type_as(fk[parent]))
                    break
            fk[lnk] = poses

        if use_names:
            return {ell.name: fk[ell] for ell in fk}
        return fk

    def visual_geometry_fk_batch(self, cfgs=None):
        """Computes the poses of the URDF's visual geometries using fk.
        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only geometries from these links will be in the returned map.
            If not specified, all links are returned.
        Returns
        -------
        fk : dict
            A map from :class:`Geometry` objects that are part of the visual
            elements of the specified links to the 4x4 homogenous transform
            matrices that position them relative to the base link's frame.
        """
        lfk = self.link_fk_batch(cfgs=cfgs)

        fk = OrderedDict()
        for link in lfk:
            for visual in link.visuals:
                fk[visual.geometry] = torch.matmul(
                    lfk[link], visual.origin.type_as(lfk[link])
                )
        return fk



import numpy as np
import torch
import random
import trimesh
from pathlib import Path
import open3d as o3d
from typing import Sequence, Union
from geometrout.primitive import Cuboid, Cylinder, Sphere
# from isaacgymenvs.utils.geometry import ObjaMesh
# from isaacgymenvs.utils.torch_urdf import TorchURDF


def construct_mixed_point_cloud(
    # obstacles: Sequence[Union[Sphere, Cuboid, Cylinder, ObjaMesh]],
    obstacles: Sequence[Union[Sphere, Cuboid, Cylinder]],
    num_points: int,
    return_point_list: bool = False,
    even: bool = False,
) -> np.ndarray:
    """
    Creates a random point cloud from a collection of obstacles. The points in
    the point cloud should be fairly(-ish) distributed amongst the obstacles based
    on their surface area.

    :param obstacles Sequence[Union[Sphere, Cuboid, Cylinder]]: The obstacles in the scene
    :param num_points int: The total number of points in the samples scene (not
                           the number of points per obstacle)
    :rtype np.ndarray: Has dim [N, 3] where N is num_points
    """
    point_set = []
    total_obstacles = len(obstacles)
    if total_obstacles == 0:
        return np.array([[]])

    # Allocate points based on obstacle surface area for even sampling
    surface_areas = np.array([o.surface_area for o in obstacles])
    total_area = np.sum(surface_areas)
    if even:
        proportions = np.ones(total_obstacles) / total_obstacles
    else:
        proportions = (surface_areas / total_area).tolist()

    indices = list(range(1, total_obstacles + 1))
    random.shuffle(indices)
    idx = 0

    for o, prop in zip(obstacles, proportions):
        sample_number = int(prop * num_points) + 500
        samples = o.sample_surface(sample_number)
        _points = indices[idx] * np.ones((sample_number, 4))
        _points[:, :3] = samples
        point_set.append(_points)
        idx += 1

    if return_point_list:
        lengths = torch.tensor([ps.shape[0] for ps in point_set], dtype=torch.float32)
        total = lengths.sum()
        ratios = lengths / total
        num_samples = (ratios * num_points).floor().to(torch.int32)
        num_samples[-1] += num_points - num_samples.sum()

        downsampled_point_set = []
        for point_subset, n in zip(point_set, num_samples):
            indices = torch.randperm(point_subset.shape[0])[:n]
            downsampled_point_set.append(point_subset[indices])

        assert (
            torch.tensor([ps.shape[0] for ps in downsampled_point_set], dtype=torch.float32).sum()
            == num_points
        )
        return downsampled_point_set

    points = np.concatenate(point_set, axis=0)

    # Downsample to the desired number of points
    return points[np.random.choice(points.shape[0], num_points, replace=False), :]


# def compute_scene_oracle_pcd(
#     num_obstacle_points: int,
#     cuboid_dims: np.ndarray = [],
#     cuboid_centers: np.ndarray = [],
#     cuboid_quats: np.ndarray = [],
#     cylinder_radii: np.ndarray = [],
#     cylinder_heights: np.ndarray = [],
#     cylinder_centers: np.ndarray = [],
#     cylinder_quats: np.ndarray = [],
#     sphere_centers: np.ndarray = [],
#     sphere_radii: np.ndarray = [],
#     mesh_position: np.ndarray = [],
#     mesh_scale: np.ndarray = [],
#     mesh_quaternion: np.ndarray = [],
#     obj_id: np.ndarray = [],
#     mesh_id: np.ndarray = [],
#     meshes_dir: str = None,
#     return_point_list: bool = False,
#     even: bool = False,
# ):
#     """
#     Compute the oracle point cloud. Input quaternions are in xyzw format
#     """

#     def quaternions_xyzw_to_wxyz(quaternions_xyzw):
#         quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]]
#         return quaternions_wxyz

#     cuboids = []
#     cylinders = []
#     spheres = []
#     meshes = []

#     if len(cuboid_dims) > 0:
#         cuboids = [
#             Cuboid(c, d, q)
#             for c, d, q in zip(cuboid_centers, cuboid_dims, quaternions_xyzw_to_wxyz(cuboid_quats))
#         ]
#         cuboids = [c for c in cuboids if not c.is_zero_volume()]

#     if len(cylinder_radii) > 0:
#         cylinders = [
#             Cylinder(c, r, h, q)
#             for c, r, h, q in zip(
#                 cylinder_centers,
#                 cylinder_radii,
#                 cylinder_heights,
#                 quaternions_xyzw_to_wxyz(cylinder_quats),
#             )
#         ]
#         cylinders = [c for c in cylinders if not c.is_zero_volume()]

#     if len(sphere_centers) > 0:
#         spheres = [Sphere(c, r) for c, r in zip(sphere_centers, sphere_radii)]
#         spheres = [s for s in spheres if not s.is_zero_volume()]

#     if len(mesh_position) > 0:
#         meshes = [
#             ObjaMesh(pos, scale, quat, obj_id, str(int(mesh_id)), meshes_dir=meshes_dir)
#             for pos, scale, quat, obj_id, mesh_id in zip(
#                 mesh_position,
#                 mesh_scale,
#                 quaternions_xyzw_to_wxyz(mesh_quaternion),
#                 obj_id,
#                 mesh_id,
#             )
#             if obj_id != 0.0 and scale > 0
#         ]
#         meshes = [m for m in meshes if not m.is_zero_volume()]

#     obstacle_points = construct_mixed_point_cloud(
#         cuboids + cylinders + spheres + meshes,
#         num_obstacle_points,
#         return_point_list=return_point_list,
#         even=even,
#     )
#     obstacle_points = np.array(obstacle_points)[..., :3]
#     return obstacle_points


def decompose_pcd_params_obs(pcd_params_obs):
    """
    Decompose the pcd params observation.
    """
    current_joint_angles, goal_joint_angles, scene_pcd_params = np.split(pcd_params_obs, [7, 14])
    goal_joint_angles, gripper_state = np.split(goal_joint_angles, [7])
    return (
        current_joint_angles,
        goal_joint_angles,
        gripper_state,
        *decompose_scene_pcd_params_obs(scene_pcd_params),
    )


def decompose_scene_pcd_params_obs(scene_pcd_params):
    """
    Decompose the pcd params observation.
    """
    M = int(scene_pcd_params[0])

    cuboid_params = scene_pcd_params[1 : 1 + 10 * M]
    cuboid_dims = cuboid_params[: 3 * M].reshape(-1, 3)
    cuboid_centers = cuboid_params[3 * M : 6 * M].reshape(-1, 3)
    cuboid_quats = cuboid_params[6 * M : 10 * M].reshape(-1, 4)

    cylinder_params = scene_pcd_params[1 + 10 * M : 1 + 10 * M + 9 * M]
    cylinder_radii = cylinder_params[: 1 * M].reshape(-1)
    cylinder_heights = cylinder_params[1 * M : 2 * M].reshape(-1)
    cylinder_centers = cylinder_params[2 * M : 5 * M].reshape(-1, 3)
    cylinder_quats = cylinder_params[5 * M : 9 * M].reshape(-1, 4)

    sphere_params = scene_pcd_params[1 + 10 * M + 9 * M : 1 + 10 * M + 9 * M + 4 * M]
    sphere_centers = sphere_params[: 3 * M].reshape(-1, 3)
    sphere_radii = sphere_params[3 * M : 4 * M].reshape(-1)

    mesh_params = scene_pcd_params[1 + 10 * M + 9 * M + 4 * M :]
    mesh_positions = mesh_params[: 3 * M].reshape(-1, 3)
    mesh_scales = mesh_params[3 * M : 4 * M].reshape(-1)
    mesh_quaternions = mesh_params[4 * M : 8 * M].reshape(-1, 4)
    obj_ids = mesh_params[8 * M : 9 * M].reshape(-1)
    mesh_ids = mesh_params[9 * M : 10 * M].reshape(-1)

    return (
        np.array(cuboid_dims).astype(np.float32),
        np.array(cuboid_centers).astype(np.float32),
        np.array(cuboid_quats).astype(np.float32),
        np.array(cylinder_radii).astype(np.float32),
        np.array(cylinder_heights).astype(np.float32),
        np.array(cylinder_centers).astype(np.float32),
        np.array(cylinder_quats).astype(np.float32),
        np.array(sphere_centers).astype(np.float32),
        np.array(sphere_radii).astype(np.float32),
        np.array(
            mesh_positions,
        ).astype(np.float32),
        np.array(
            mesh_scales,
        ).astype(np.float32),
        np.array(
            mesh_quaternions,
        ).astype(np.float32),
        np.array(
            obj_ids,
        ).astype(np.float32),
        np.array(
            mesh_ids,
        ).astype(np.float32),
    )


def quaternion_to_rotation_matrix(q):
    # q: (..., 4) -> (..., 3, 3)
    # (qx, qy, qz, qw)
    x, y, z, w = q.unbind(-1)

    B = q.shape[:-1]

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = torch.stack([
        ww + xx - yy - zz, 2 * (xy - wz),       2 * (xz + wy),
        2 * (xy + wz),     ww - xx + yy - zz,   2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),       ww - xx - yy + zz
    ], dim=-1).reshape(*B, 3, 3)
    return rot


def transform_pcds_to_world(pcds_local, poses):
    # pcds_local: (N, D, P, 3)
    # poses: (N, D, 7) -> (x, y, z, qx, qy, qz, qw)
    trans = poses[..., :3]  # (..., 3)
    quat = poses[..., 3:]   # (..., 4)

    rot = quaternion_to_rotation_matrix(quat).to(pcds_local.dtype)  # (..., 3, 3)

    # Transform pointclouds
    pcds_local = pcds_local.unsqueeze(-1)  # (..., P, 3, 1)
    pcds_rotated = torch.matmul(rot.unsqueeze(-3), pcds_local).squeeze(-1)  # (..., P, 3)
    pcds_world = pcds_rotated + trans.unsqueeze(-2)  # (N, D, P, 3)
    return pcds_world


def shuffle_pcd(pcd: torch.Tensor) -> torch.Tensor:
    """
    Randomize ordering of points in a point cloud.
    
    Args:
        pcd: (B, N, 3) tensor
    Returns:
        shuffled: (B, N, 3) tensor with points randomly permuted
    """
    B, N, _ = pcd.shape
    device = pcd.device

    # Random permutations (different per batch)
    idx = torch.argsort(torch.rand(B, N, device=device), dim=-1)  # (B, N)

    # Build batch index for gather
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)

    # Gather shuffled points
    return pcd[batch_idx, idx]  # (B, N, 3)


def crop_local_pcd(pcd: torch.Tensor, local_range: torch.float, num_local_points: torch.int):
    """
    Crop the point cloud to a local region around the origin with 0 padding.
    Args:
        pcd: (B, N, 3) tensor
        range: float, the radius of the local region
    """
    B, N, _ = pcd.shape
    device = pcd.device

    # get local pcd
    masked_pcds = shuffle_pcd(pcd)
    dist = torch.norm(masked_pcds, dim=-1)
    mask = dist < local_range # nan < X always returns false, so if there are nan values in pcd input, it get automatically filtered out
    masked_pcds[~mask] = float("nan")

    # sort to get all the valid points
    is_valid = mask.int()
    sort_idx = torch.argsort(is_valid, dim=-1, descending=True)
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)
    sorted_pcds = masked_pcds[batch_idx, sort_idx]  # (B, N, 3)
    pcd_local_nan_padding = sorted_pcds[:, :num_local_points]

    avg_num_valid_points = is_valid.sum() / B
    min_num_valid_points = is_valid.sum(dim=-1).min()
    logs = {
        "local_crop/avg_num_valid_points": avg_num_valid_points.item(),
        "local_crop/min_num_valid_points": min_num_valid_points.item(),
    }

    # replace nan values as 0s
    local_pcd_zero_padding = torch.nan_to_num(pcd_local_nan_padding, nan=0.0)

    return local_pcd_zero_padding, logs


def transform_pointcloud(pc, T):
    """pc: (B,N,3), T: (B,4,4) -> (B,N,3)"""
    B, N, _ = pc.shape
    homo = torch.cat([pc, torch.ones(B, N, 1, device=pc.device)], dim=-1)  # (B,N,4)
    out = torch.matmul(T, homo.transpose(1,2))  # (B,4,N)
    return out[:, :3].transpose(1,2)  # (B,N,3)


def visualize_pcd(points):
    """
    points: (N,3) torch or numpy
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.2, 0.6, 0.9])  # light blue
    o3d.visualization.draw_geometries([pcd])

import re
def resolve_mesh_path(urdf_path: str, mesh_filename: str):
    urdf_dir = Path(urdf_path).parent
    match = re.match(r'^package:/{1,2}(.+)$', mesh_filename.strip())
    if match:
        # It's a package URI → join the extracted relative path to urdf_dir
        rel_path = match.group(1)
        return urdf_dir.parent / rel_path
    else:
        # Plain relative/absolute path → resolve normally
        return urdf_dir / mesh_filename

class FrankaLeapSampler:
    def __init__(self, urdf_path, device = "cuda", num_points=4096):
        self.device = device
        self.urdf_path = urdf_path
        self.robot = TorchURDF.load(urdf_path, lazy_load_meshes=True, device=device)
        # Load meshes for all links with visuals
        self.links = [l for l in self.robot.links if len(l.visuals) and (l.visuals[0].geometry.mesh is not None)]
        self.hand_links = [l for l in self.links if ("panda" not in l.name) and l.visuals[0].geometry.mesh is not None]

        # meshes = [
        #     trimesh.load(resolve_mesh_path(urdf_path, l.visuals[0].geometry.mesh.filename), force="mesh")
        #     for l in self.links
        # ]

        meshes = []
        for l in self.links:
            mesh = None
            for v in l.visuals:
                if mesh is None:
                    mesh = trimesh.load(resolve_mesh_path(urdf_path, v.geometry.mesh.filename), force="mesh")
                else:
                    mesh += trimesh.load(resolve_mesh_path(urdf_path, v.geometry.mesh.filename), force="mesh")
            meshes.append(mesh)

        areas = np.array([m.bounding_box_oriented.area for m in meshes])
        n_pts = np.round(num_points * areas / areas.sum()).astype(int)
        n_pts[0] += num_points - n_pts.sum()  # fix rounding
        self.points = {
            l.name: torch.as_tensor(
                trimesh.sample.sample_surface(meshes[i], n_pts[i])[0],
                device=device, dtype=torch.float32
            ).unsqueeze(0)  # (1,Ni,3)
            for i, l in enumerate(self.links)
        }

    def sample(self, joint_angles, joint_mapping_list=None, num_points=None, hand_only=False):
        """
        joint_angles: (B, 23) joint config
        joint_mapping_list: list[int], optional mapping to torch_urdf ordering
        hand_only: if True, only sample from hand_links
        returns: (B, num_points, 3) world-frame pointcloud
        """
        if joint_angles.ndim == 1:
            joint_angles = joint_angles.unsqueeze(0)

        if joint_mapping_list is not None:
            joint_angles = joint_angles[:, joint_mapping_list]

        fk = self.robot.visual_geometry_fk_batch(joint_angles)  # dict[geom] -> (B,4,4)
        pcs = []

        link_set = self.hand_links if hand_only else self.links
        for l in link_set:
            T = fk[l.visuals[0].geometry]  # (B,4,4)
            pc = self.points[l.name].repeat(joint_angles.shape[0], 1, 1)  # (B,Ni,3)
            pcs.append(transform_pointcloud(pc, T))

        pc = torch.cat(pcs, dim=1)  # (B, totalN, 3)
        if num_points is None:
            return pc
        idx = np.random.choice(pc.shape[1], num_points, replace=False)
        return pc[:, idx, :]

sampler = None

def sample_pointcloud(urdf_path, joint_angles, device = "cuda", verbose = False):
    global sampler
    if sampler is None or sampler.urdf_path != urdf_path:
        sampler = FrankaLeapSampler(urdf_path, device)
    pcd = sampler.sample(joint_angles)

    if verbose:
        tensor_to_ply(pcd[0], "pointcloud.ply")
        print(f"Saved {pcd[0].shape[0]} points to pointcloud.ply")
    return pcd

if __name__ == "__main__":
    urdf_path = "/home/glorbo4/peiqi/DoorOpening/source/DoorOpening/assets/door/PartNet/8893/mobility.urdf"
    # urdf_path = "/home/glorbo4/peiqi/DoorOpening/source/DoorOpening/assets/glorbot/glorbot.urdf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = FrankaLeapSampler(urdf_path, device)
    joint_angles = torch.randn(1, 32, device=device)
    pcd = sampler.sample(joint_angles)
    verbose = True
    if verbose:
        from DoorOpening.utils.point_utils import tensor_to_ply
        tensor_to_ply(pcd[0], "points.ply")
    print(pcd.shape)