from typing import List
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Calib:
    K: np.ndarray  # 3x3
    Rt: np.ndarray  # 3x4
    P: np.ndarray  # 3x4
    Kr_inv: np.ndarray  # 3x3

    @property
    def cam_loc(self):
        return -self.Rt[:3, :3].T @ self.Rt[:3, 3]


def unproject_uv_to_rays(points: np.ndarray, calib: Calib):
    """
    Parameters
    ----------
    points: Nx2
    calib:
    """
    points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    rays = (calib.Kr_inv @ points.T).T
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def points_to_lines_distances(points_0: np.ndarray, points_1: np.ndarray, rays_1: np.ndarray):
    # TODO: vectorize the code
    n_points = len(points_0)
    dsts = []
    for i in range(n_points):
        p0, p1 = points_0[i, :3], points_1[i, :3]
        dst = np.linalg.norm(np.cross(p0-p1, rays_1[i, :3]))
        dsts.append(dst)
    return dsts


def lines_to_lines_distance(points_0: np.ndarray, rays_0: np.ndarray, points_1: np.ndarray, rays_1: np.ndarray):
    # TODO: vectorize the code
    n_points = len(points_0)
    dsts = []
    for i in range(n_points):
        p0, p1 = points_0[i, :3], points_1[i, :3]
        ray0, ray1 = rays_0[i, 3:], rays_1[i, 3:]
        if np.dot(ray0, ray1) < 1e-5:
            # parallel rays
            dst = np.linalg.norm(np.cross(p0 - p1, ray0))
        else:
            n = np.cross(ray0, ray1)
            n = n / np.linalg.norm(n)
            dst = np.abs(np.dot(p0 - p1, n))
        dsts.append(dst)
    return dsts


def line_to_point_distance(a, b, c, x, y):
    return abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)


def get_fundamental_matrix(p1, p2):
    """
    adapted from https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp
    :param p1:
    :param p2:
    :return:
    """
    x = [np.vstack([p1[1, :], p1[2, :]]),
         np.vstack([p1[2, :], p1[0, :]]),
         np.vstack([p1[0, :], p1[1, :]])]

    y = [np.vstack([p2[1, :], p2[2, :]]),
         np.vstack([p2[2, :], p2[0, :]]),
         np.vstack([p2[0, :], p2[1, :]])]

    f_mat = np.zeros((3, 3), dtype=p1.dtype)
    for i in range(3):
        for j in range(3):
            xy = np.vstack([x[j], y[i]])
            f_mat[i, j] = np.linalg.det(xy)
    return f_mat


def line_to_point_distance(a, b, c, x, y):
    return abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)


def calc_epipolar_error(cam1: Calib, keypoints_1: np.ndarray, scores_1: np.ndarray,
                        cam2: Calib, keypoints_2: np.ndarray, scores_2: np.ndarray,
                        score_weighted: bool):
    f_mat = get_fundamental_matrix(cam1.P, cam2.P)
    n_joint = len(keypoints_1)

    if len(keypoints_1) == 0:
        return np.finfo(np.float32).max

    epilines_1to2 = cv2.computeCorrespondEpilines(keypoints_1.reshape((-1, 1, 2)), 1, f_mat)
    epilines_1to2 = epilines_1to2.reshape((-1, 3))

    epilines_2to1 = cv2.computeCorrespondEpilines(keypoints_2.reshape((-1, 1, 2)), 2, f_mat)
    epilines_2to1 = epilines_2to1.reshape((-1, 3))

    invalid_mask = np.isclose(scores_1 * scores_2, 0.0).flatten()
    if not score_weighted:
        # bad or good, valid or invalid has the same weight on the total cost
        kps_cost_factor = np.ones(len(scores_1))
    else:
        # TODO: what happens if two mismatches have perfect confidence score?
        #  In this case, their kps_cost_factor will be 0 and they still have low error
        # higher average confidence score means this pair will lower the Epipolar distance [the cost].
        # lower average confidence score means this pair will increase the Epipolar distance [the cost]
        kps_cost_factor = 1.0 - 0.5 * (scores_1 + scores_2)
        # for avoiding the effect of invalid key-points whose confidences are zero
        kps_cost_factor[invalid_mask] = 0.0

    # kps_cost_factor = np.ones(len(scores_1))
    # kps_cost_factor[invalid_mask] = 0.0

    if np.all(invalid_mask):
        INFTY_COST = 1e+5
        total = INFTY_COST
    else:
        total = 0
        for i in range(n_joint):
            p1 = keypoints_1[i, :]
            p2 = keypoints_2[i, :]
            l1to2 = epilines_1to2[i, :]
            l2to1 = epilines_2to1[i, :]
            d1 = line_to_point_distance(*l1to2, *p2)
            d2 = line_to_point_distance(*l2to1, *p1)
            total = total + (d1 + d2) * kps_cost_factor[i]

        total_score = max(float(np.sum(kps_cost_factor)), 1e-5)
        total = total / total_score  # normalize

    return total


def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous

    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def triangulate_point_groups_from_multiple_views_linear(proj_matricies: np.ndarray,
                                                        points_grps: List[np.ndarray],
                                                        min_score):
    """
    :param min_score: just apply triangulation on keypoints whose scores greater than this value. if no valid keypoints
    exist, resort to all keypoints
    :param proj_matricies: (N,3,4): sequence of projection matrices
    :param points_grps: List[Nx3]
    """
    n_kps = len(points_grps[0])
    kps_3ds = []
    for kps_idx in range(n_kps):
        points = []
        cams_p = []
        for grp_idx, grp in enumerate(points_grps):
            if grp[kps_idx, 2] >= min_score:
                points.append(grp[kps_idx, :])
                cams_p.append(proj_matricies[grp_idx, :])

        if len(points) < 2:
            # if not enough points are collected. resort to all points
            points = np.array([grp[kps_idx, :] for grp in points_grps])
            score = np.mean(np.array([grp[kps_idx, 2] for grp in points_grps]))
            cams_p = proj_matricies
        else:
            points = np.array(points)
            cams_p = np.array(cams_p)
            score = np.mean(points[:, 2])

        p = triangulate_point_from_multiple_views_linear(cams_p, points[:, :2])
        kps_3ds.append((p[0], p[1], p[2], score))

    return np.array(kps_3ds)


def triangulate_point_from_multiple_views_linear(proj_matricies: np.ndarray, points: np.ndarray):
    """Triangulates one point from multiple (N) views using direct linear transformation (DLT).
    For more information look at "Multiple view geometry in computer vision",
    Richard Hartley and Andrew Zisserman, 12.2 (p. 312).

    Args:
        proj_matricies numpy array of shape (N, 3, 4): sequence of projection matricies (3x4)
        points numpy array of shape (N, 2): sequence of points' coordinates

    Returns:
        point_3d numpy array of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)
    a_mat = np.zeros((2 * n_views, 4))
    for j in range(len(proj_matricies)):
        a_mat[j * 2 + 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
        a_mat[j * 2 + 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]

    u, s, vh = np.linalg.svd(a_mat, full_matrices=False)
    point_3d_homo = vh[3, :]

    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d


def project_3d_points_to_image_plane_without_distortion(proj_matrix, points_3d, convert_back_to_euclidean=True):
    """Project 3D points to image plane not taking into account distortion
    Args:
        proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
        points_3d numpy array or torch tensor of shape (N, 3): 3D points
        convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                        NOTE: division by zero can be here if z = 0
    Returns:
        numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")
