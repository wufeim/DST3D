import math

import numpy as np
from scipy.linalg import logm


def get_transformation_matrix(azimuth, elevation, distance):
    if distance == 0:
        # return None
        distance = 0.1

    # camera center
    C = np.zeros((3, 1))
    C[0] = distance * math.cos(elevation) * math.sin(azimuth)
    C[1] = -distance * math.cos(elevation) * math.cos(azimuth)
    C[2] = distance * math.sin(elevation)

    # rotate coordinate system by theta is equal to rotating the model by theta
    azimuth = -azimuth
    elevation = -(math.pi / 2 - elevation)

    # rotation matrix
    Rz = np.array(
        [
            [math.cos(azimuth), -math.sin(azimuth), 0],
            [math.sin(azimuth), math.cos(azimuth), 0],
            [0, 0, 1],
        ]
    )  # rotation by azimuth
    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(elevation), -math.sin(elevation)],
            [0, math.sin(elevation), math.cos(elevation)],
        ]
    )  # rotation by elevation
    R_rot = np.dot(Rx, Rz)
    R = np.hstack((R_rot, np.dot(-R_rot, C)))
    R = np.vstack((R, [0, 0, 0, 1]))

    return R


def rotation_theta(theta):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def cal_rotation_matrix(theta, elev, azum, dis):
    if dis <= 1e-10:
        dis = 0.5
    return rotation_theta(theta) @ get_transformation_matrix(azum, elev, dis)[0:3, 0:3]


def pose_error(gt, pred):
    if pred is None:
        return np.pi
    azimuth_gt, elevation_gt, theta_gt = (
        float(gt["azimuth"]),
        float(gt["elevation"]),
        float(gt["theta"]),
    )
    azimuth_pred, elevation_pred, theta_pred = (
        float(pred["azimuth"]),
        float(pred["elevation"]),
        float(pred["theta"]),
    )
    anno_matrix = cal_rotation_matrix(theta_gt, elevation_gt, azimuth_gt, 5.0)
    pred_matrix = cal_rotation_matrix(theta_pred, elevation_pred, azimuth_pred, 5.0)
    if (
        np.any(np.isnan(anno_matrix))
        or np.any(np.isnan(pred_matrix))
        or np.any(np.isinf(anno_matrix))
        or np.any(np.isinf(pred_matrix))
    ):
        error_ = np.pi
    else:
        error_ = (
            (logm(np.dot(np.transpose(pred_matrix), anno_matrix)) ** 2).sum()
        ) ** 0.5 / (2.0 ** 0.5)
    return error_
