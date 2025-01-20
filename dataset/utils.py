import numpy as np
from pyquaternion import Quaternion
import torch


def get_rm(angle, axis, deg=False):
    if deg:
        angle = np.deg2rad(angle)
    rm = np.eye(3)
    if axis == 'x':
        rm[1, 1] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[1, 2] = - np.sin(angle)
        rm[2, 1] = np.sin(angle)
    elif axis == 'y':
        rm[0, 0] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[0, 2] = np.sin(angle)
        rm[2, 0] = - np.sin(angle)
    elif axis == 'z':
        rm[0, 0] = np.cos(angle)
        rm[1, 1] = np.cos(angle)
        rm[0, 1] = - np.sin(angle)
        rm[1, 0] = np.sin(angle)
    return rm


def get_xyz(pose_dict):
    return np.array(pose_dict['translation'])

def get_img2global(calib_dict, pose_dict):
    
    cam2img = np.eye(4)
    cam2img[:3, :3] = np.asarray(calib_dict['camera_intrinsic'])
    img2cam = np.linalg.inv(cam2img)

    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    img2global = ego2global @ cam2ego @ img2cam
    return img2global

def get_lidar2global(calib_dict, pose_dict):

    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    lidar2global = ego2global @ lidar2ego
    return lidar2global


def get_lidar2cam(cam_calib_dict, cam_pose_dict, lidar2global):
    """Get the transformation matrix from lidar to camera

    Args:
        cam_calib_dict (dict): contains the camera to ego transformation
        cam_pose_dict (dict): contains the ego to global transformation
        lidar2global (np.ndarray): the transformation matrix from lidar to global

    Returns:
        np.ndarray: the transformation matrix from lidar to camera
    """
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(cam_calib_dict['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.asarray(cam_calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(cam_pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(cam_pose_dict['translation']).T

    cam2global = ego2global @ cam2ego

    lidar2cam = np.linalg.inv(cam2global) @ lidar2global
    return lidar2cam


def custom_collate_fn_temporal(instances):
    return_dict = {}
    for k, v in instances[0].items():
        if isinstance(v, np.ndarray):
            return_dict[k] = torch.stack([
                torch.from_numpy(instance[k]) for instance in instances])
        elif isinstance(v, torch.Tensor):
            return_dict[k] = torch.stack([instance[k] for instance in instances])
        elif isinstance(v, (dict, str)):
            return_dict[k] = [instance[k] for instance in instances]
        elif v is None:
            return_dict[k] = [None] * len(instances)
        else:
            raise NotImplementedError
    return return_dict
