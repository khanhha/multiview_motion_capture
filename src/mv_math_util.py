from typing import List, Tuple
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import least_squares
import torch


@dataclass
class Calib:
    K: np.ndarray  # 3x3
    Rt: np.ndarray  # 3x4
    P: np.ndarray  # 3x4
    Kr_inv: np.ndarray  # 3x3
    img_wh_size: Tuple[int, int]

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
        dst = np.linalg.norm(np.cross(p0 - p1, rays_1[i, :3]))
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


def calc_epipolar_error(cam1: Calib, keypoints_1: np.ndarray, scores_1: np.ndarray,
                        cam2: Calib, keypoints_2: np.ndarray, scores_2: np.ndarray,
                        min_valid_kps_score=0.05, invalid_default_error=np.nan):
    f_mat = get_fundamental_matrix(cam1.P, cam2.P)
    n_joint = len(keypoints_1)

    if len(keypoints_1) == 0:
        return invalid_default_error

    epilines_1to2 = cv2.computeCorrespondEpilines(keypoints_1.reshape((-1, 1, 2)), 1, f_mat)
    epilines_1to2 = epilines_1to2.reshape((-1, 3))

    epilines_2to1 = cv2.computeCorrespondEpilines(keypoints_2.reshape((-1, 1, 2)), 2, f_mat)
    epilines_2to1 = epilines_2to1.reshape((-1, 3))

    valid_mask = (scores_1 * scores_2).flatten() > min_valid_kps_score

    if np.any(valid_mask):
        total = 0
        cnt = 0
        for i in range(n_joint):
            if not valid_mask[i]:
                continue
            p1 = keypoints_1[i, :]
            p2 = keypoints_2[i, :]
            l1to2 = epilines_1to2[i, :]
            l2to1 = epilines_2to1[i, :]
            d1 = line_to_point_distance(*l1to2, *p2)
            d2 = line_to_point_distance(*l2to1, *p1)
            total = total + 0.5 * (d1 + d2)
            cnt += 1
        total = total / cnt

        return total
    else:
        return invalid_default_error


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


def triangulate_point_groups_from_multiple_views_linear(proj_matricies: List[np.ndarray],
                                                        points_grps: List[np.ndarray],
                                                        min_score,
                                                        post_optimize=False,
                                                        n_max_iter=2):
    """
    :param n_max_iter:
    :param min_score: just apply triangulation on keypoints whose scores greater than this value. if no valid keypoints
    exist, resort to all keypoints
    :param proj_matricies: List[(3,4)]: sequence of projection matrices
    :param points_grps: List[Nx3]
    :param post_optimize:
    """
    n_kps = len(points_grps[0])
    kps_3ds = []
    for kps_idx in range(n_kps):
        points_2d = []
        cams_p = []
        for grp_idx, grp in enumerate(points_grps):
            if grp[kps_idx, 2] >= min_score:
                points_2d.append(grp[kps_idx, :])
                cams_p.append(proj_matricies[grp_idx])

        if len(points_2d) < 2:
            # if not enough points are collected. resort to all points
            points_2d = np.array([grp[kps_idx, :] for grp in points_grps])
            score = np.mean(np.array([grp[kps_idx, 2] for grp in points_grps]))
            cams_p = proj_matricies
        else:
            points_2d = np.array(points_2d)
            score = np.mean(points_2d[:, 2])

        point_3d = triangulate_point_from_multiple_views_linear(cams_p, points_2d[:, :2])
        kps_3ds.append((point_3d[0], point_3d[1], point_3d[2], score))

    kps_3ds = np.array(kps_3ds)

    if post_optimize:
        n_cams = len(proj_matricies)

        def _residual_func(_x):
            _joint_locs = _x.reshape((-1, 3))
            _joint_homos = np.concatenate([_joint_locs, np.ones((_joint_locs.shape[0], 1))], axis=-1).T
            _diff_reprojs = []
            for _vi in range(n_cams):
                _proj = proj_matricies[_vi] @ _joint_homos
                _proj = (_proj[:2] / (_proj[2] + 1e-6)).T
                _d = np.linalg.norm(_proj - points_grps[_vi][:, :2], axis=-1)
                _d = _d * points_grps[_vi][:, -1]
                _diff_reprojs.append(_d)
            _diff_reprojs = np.array(_diff_reprojs).flatten()
            return _diff_reprojs

        try:
            params = kps_3ds[:, :3].flatten().copy()
            res = least_squares(_residual_func, params, max_nfev=n_max_iter)
            kps_3ds[:, :3] = res.x.reshape((-1, 3))
        except Exception as exp:
            print(exp)

    return kps_3ds


def triangulate_point_from_multiple_views_linear(proj_matricies: List[np.ndarray], points: np.ndarray):
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


def calc_pairwise_f_mats(calibs: List[Calib]):
    skew_op = lambda x: torch.tensor([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: torch.inverse(K_0).t() @ (
            R_0 @ R_1.t()) @ K_1.t() @ skew_op(K_1 @ R_1 @ R_0.t() @ (T_0 - R_0 @ R_1.t() @ T_1))

    fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: fundamental_op(K_0, RT_0[:, :3], RT_0[:, 3], K_1,
                                                                    RT_1[:, :3], RT_1[:, 3])
    F = torch.zeros(len(calibs), len(calibs), 3, 3)  # NxNx3x3 matrix
    # TODO: optimize this stupid nested for loop
    for i in range(len(calibs)):
        for j in range(len(calibs)):
            F[i, j] += fundamental_RT_op(torch.tensor(calibs[i].K),
                                         torch.tensor(calibs[i].Rt),
                                         torch.tensor(calibs[j].K), torch.tensor(calibs[j].Rt))
            if F[i, j].sum() == 0:
                F[i, j] += 1e-12  # to avoid nan

    return F.numpy()


def projected_distance(pts_0, pts_1, F):
    """
    Compute point distance with epipolar geometry knowledge
    :param pts_0: numpy points array with shape Nx17x2
    :param pts_1: numpy points array with shape Nx17x2
    :param F: Fundamental matrix F_{01}
    :return: numpy array of pairwise distance
    """
    # lines = cv2.computeCorrespondEpilines ( pts_0.reshape ( -1, 1, 2 ), 2,
    #                                         F )  # I know 2 is not seems right, but it actually work for this dataset
    # lines = lines.reshape ( -1, 3 )
    # points_1 = np.ones ( (lines.shape[0], 3) )
    # points_1[:, :2] = pts_1.reshape((-1, 2))
    #
    # # to begin here!
    # dist = np.sum ( lines * points_1, axis=1 ) / np.linalg.norm ( lines[:, :2], axis=1 )
    # dist = np.abs ( dist )
    # dist = np.mean ( dist )

    lines = cv2.computeCorrespondEpilines(pts_0.reshape(-1, 1, 2), 2, F)
    lines = lines.reshape(-1, 17, 1, 3)
    lines = lines.transpose(0, 2, 1, 3)
    points_1 = np.ones([1, pts_1.shape[0], 17, 3])
    points_1[0, :, :, :2] = pts_1

    dist = np.sum(lines * points_1, axis=3)  # / np.linalg.norm(lines[:, :, :, :2], axis=3)
    dist = np.abs(dist)
    dist = np.mean(dist, axis=2)

    return dist


def geometry_affinity(points_set, Fs, dimGroup):
    M, _, _ = points_set.shape
    # distance_matrix = np.zeros ( (M, M), dtype=np.float32 )
    distance_matrix = np.ones((M, M), dtype=np.float32) * 50
    np.fill_diagonal(distance_matrix, 0)
    # TODO: remove this stupid nested for loop
    import time
    start_time = time.time()
    n_groups = len(dimGroup)
    for cam_id0, h in enumerate(range(n_groups - 1)):
        for cam_add, k in enumerate(range(cam_id0 + 1, n_groups - 1)):
            cam_id1 = cam_id0 + cam_add + 1
            # if there is no one in some view, skip it!
            if dimGroup[h] == dimGroup[h + 1] or dimGroup[k] == dimGroup[k + 1]:
                continue

            pose_id0 = points_set[dimGroup[h]:dimGroup[h + 1]]
            pose_id1 = points_set[dimGroup[k]:dimGroup[k + 1]]
            mean_dst = 0.5 * (projected_distance(pose_id0, pose_id1, Fs[cam_id0, cam_id1]) +
                              projected_distance(pose_id1, pose_id0, Fs[cam_id1, cam_id0]).T)
            distance_matrix[dimGroup[h]:dimGroup[h + 1], dimGroup[k]:dimGroup[k + 1]] = mean_dst
            # symmetric matrix
            distance_matrix[dimGroup[k]:dimGroup[k + 1], dimGroup[h]:dimGroup[h + 1]] = \
                distance_matrix[dimGroup[h]:dimGroup[h + 1], dimGroup[k]:dimGroup[k + 1]].T

    end_time = time.time()
    # print('using %fs' % (end_time - start_time))

    affinity_matrix = - (distance_matrix - distance_matrix.mean()) / distance_matrix.std()
    # TODO: add flexible factor
    affinity_matrix = 1 / (1 + np.exp(-5 * affinity_matrix))
    return distance_matrix, affinity_matrix
