import os.path as osp
import sys
import time
import torch
import numpy as np
import cv2


def myproj2dpam(Y, tol=1e-4):
    X0 = Y
    X = Y
    I2 = 0

    for iter_ in range(10):

        X1 = projR(X0 + I2)
        I1 = X1 - (X0 + I2)
        X2 = projC(X0 + I1)
        I2 = X2 - (X0 + I1)

        chg = torch.sum(torch.abs(X2[:] - X[:])) / X.numel()
        X = X2
        if chg < tol:
            return X
    return X


def projR(X):
    for i in range(X.shape[0]):
        X[i, :] = proj2pav(X[i, :])
        # X[i, :] = proj2pavC ( X[i, :] )
    return X


def projC(X):
    for j in range(X.shape[1]):
        # X[:, j] = proj2pavC ( X[:, j] )
        # Change to tradition implementation
        X[:, j] = proj2pav(X[:, j])
    return X


def proj2pav(y):
    y[y < 0] = 0
    x = torch.zeros_like(y)
    if torch.sum(y) < 1:
        x += y
    else:
        u, _ = torch.sort(y, descending=True)
        sv = torch.cumsum(u, 0)
        to_find = u > (sv - 1) / (torch.arange(1, len(u) + 1, device=u.device, dtype=u.dtype))
        rho = torch.nonzero(to_find.reshape(-1))[-1]
        theta = torch.max(torch.tensor(0, device=sv.device, dtype=sv.dtype), (sv[rho] - 1) / (rho.float() + 1))
        x += torch.max(y - theta, torch.tensor(0, device=sv.device, dtype=y.dtype))
    return x


def proj2pavC(y):
    # % project an n-dim vector y to the simplex Dn
    # % Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}
    #
    # % (c) Xiaojing Ye
    # % xyex19@gmail.com
    # %
    # % Algorithm is explained as in the linked document
    # % http://arxiv.org/abs/1101.6081
    # % or
    # % http://ufdc.ufl.edu/IR00000353/
    # %
    # % Jan. 14, 2011.

    m = len(y)
    bget = False

    s, _ = torch.sort(y, descending=True)
    tmpsum = 0

    for ii in range(m - 1):
        tmpsum = tmpsum + s[ii]
        # tmax = (tmpsum - 1) / ii
        tmax = (tmpsum - 1) / (ii + 1)  # change since index starts from 0
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    x = torch.max(y - tmax, torch.zeros_like(y))
    return x


def transform_closure(X_bin):
    """
    Convert binary relation matrix to permutation matrix
    :param X_bin: torch.tensor which is binarized by a threshold
    :return:
    """
    temp = torch.zeros_like(X_bin)
    N = X_bin.shape[0]
    for k in range(N):
        for i in range(N):
            for j in range(N):
                temp[i][j] = X_bin[i, j] or (X_bin[i, k] and X_bin[k, j])
    vis = torch.zeros(N)
    match_mat = torch.zeros_like(X_bin)
    for i, row in enumerate(temp):
        if vis[i]:
            continue
        for j, is_relative in enumerate(row):
            if is_relative:
                vis[j] = 1
                match_mat[j, i] = 1
    return match_mat


def match_als(W: np.ndarray, dimGroup, **kwargs):
    """
    % This function is to solve
    % min - <W,X> + alpha||X||_* + beta||X||_1, st. X \in C

    % The problem is rewritten as
    % <beta-W,AB^T> + alpha/2||A||^2 + alpha/2||B||^2
    % st AB^T=Z, Z\in\Omega

    % ---- Output:
    % X: a sparse binary matrix indicating correspondences
    % A: AA^T = X;
    % info: other info.

    % ---- Required input:
    % W: sparse input matrix storing scores of pairwise matches
    % dimGroup: a vector storing the number of points on each objects

    % ---- Other options:
    % maxRank: the restricted rank of X* (select it as large as possible)
    % alpha: the weight of nuclear norm
    % beta: the weight of l1 norm
    % pSelect: propotion of selected points, i.e., m'/m in section 5.4 in the paper
    % tol: tolerance of convergence
    % maxIter: maximal iteration
    % verbose: display info or not
    % eigenvalues: output eigenvalues or not
    """
    # optional paramters
    alpha = 50
    beta = 0.1
    # maxRank = max(dimGroup) * 4
    n_max_pp = np.diff(dimGroup)
    maxRank = max(n_max_pp) + 2

    pSelect = 1
    tol = 5e-4
    maxIter = 1000
    verbose = False
    eigenvalues = False
    W = 0.5 * (W + W.transpose())
    X = W.copy()
    Z = W.copy()
    Y = np.zeros_like(W)
    mu = 64
    n = X.shape[0]
    maxRank = min(n, maxRank)
    A = np.random.rand(n, maxRank)

    iter_cnt = 0
    t0 = time.time()
    for iter_idx in range(maxIter):
        X0 = X.copy()
        X = Z - (Y - W + beta) / mu
        B = (np.linalg.inv(A.transpose() @ A + alpha / mu * np.eye(maxRank)) @ (A.transpose() @ X)).transpose()
        A = (np.linalg.inv(B.transpose() @ B + alpha / mu * np.eye(maxRank)) @ (
                    B.transpose() @ X.transpose())).transpose()
        X = A @ B.transpose()

        Z = X + Y / mu
        # enforce the self-matching to be null
        for i in range(len(dimGroup) - 1):
            ind1, ind2 = dimGroup[i], dimGroup[i + 1]
            Z[ind1:ind2, ind1:ind2] = 0

        if pSelect == 1:
            Z[np.arange(n), np.arange(n)] = 1

        Z[Z < 0] = 0
        Z[Z > 1] = 1

        Y = Y + mu * (X - Z)

        # test if convergence
        pRes = np.linalg.norm(X - Z) / n
        dRes = mu * np.linalg.norm(X - X0) / n
        if verbose:
            print('Iter = %d, Res = (%d,%d), mu = %d \n', iter, pRes, dRes, mu)

        if pRes < tol and dRes < tol:
            iter_cnt = iter_idx
            break

        if pRes > 10 * dRes:
            mu = 2 * mu
        elif dRes > 10 * pRes:
            mu = mu / 2

    X = 0.5 * (X + X.transpose())
    X_bin = X > 0.5

    total_time = time.time() - t0

    match_mat = transform_closure(torch.tensor(X_bin))

    return match_mat


def matchSVT(S, dimGroup, **kwargs):
    alpha = kwargs.get('alpha', 0.1)
    pSelect = kwargs.get('pselect', 1)
    tol = kwargs.get('tol', 5e-4)
    maxIter = kwargs.get('maxIter', 20)
    verbose = kwargs.get('verbose', False)
    eigenvalues = kwargs.get('eigenvalues', False)
    _lambda = kwargs.get('_lambda', 50)
    mu = kwargs.get('mu', 64)
    dual_stochastic = kwargs.get('dual_stochastic_SVT', True)
    if verbose:
        print('Running SVT-Matching: alpha = %.2f, pSelect = %.2f _lambda = %.2f \n' % (
            alpha, pSelect, _lambda))
    info = dict()
    N = S.shape[0]
    S[torch.arange(N), torch.arange(N)] = 0
    S = (S + S.t()) / 2
    X = S.clone()
    use_spectral = False
    if use_spectral:
        eig_value, eig_vector = S.eig(eigenvectors=True)
        _, eig_idx = torch.sort(eig_value[:, 0], descending=True)
        X[:, :S.shape[1]] = eig_vector.t()

    Y = torch.zeros_like(S)
    W = alpha - S
    t0 = time.time()

    n_iter_1 = maxIter

    for iter_ in range(maxIter):

        X0 = X
        # update Q with SVT
        U, s, V = torch.svd(1.0 / mu * Y + X)
        diagS = s - _lambda / mu
        diagS[diagS < 0] = 0
        Q = U @ diagS.diag() @ V.t()
        # update X
        X = Q - (W + Y) / mu
        # project X
        for i in range(len(dimGroup) - 1):
            ind1, ind2 = dimGroup[i], dimGroup[i + 1]
            X[ind1:ind2, ind1:ind2] = 0
        if pSelect == 1:
            X[torch.arange(N), torch.arange(N)] = 1
        X[X < 0] = 0
        X[X > 1] = 1

        if dual_stochastic:
            # Projection for double stochastic constraint
            for i in range(len(dimGroup) - 1):
                row_begin, row_end = int(dimGroup[i]), int(dimGroup[i + 1])
                for j in range(len(dimGroup) - 1):
                    col_begin, col_end = int(dimGroup[j]), int(dimGroup[j + 1])
                    if row_end > row_begin and col_end > col_begin:
                        X[row_begin:row_end, col_begin:col_end] = myproj2dpam(X[row_begin:row_end, col_begin:col_end],
                                                                              1e-2)

        X = (X + X.t()) / 2
        # update Y
        Y = Y + mu * (X - Q)
        # test if convergence
        pRes = torch.norm(X - Q) / N
        dRes = mu * torch.norm(X - X0) / N
        if verbose:
            print(f'Iter = {iter_}, Res = ({pRes}, {dRes}), mu = {mu}')

        if pRes < tol and dRes < tol:
            n_iter_1 = iter_
            break

        if pRes > 10 * dRes:
            mu = 2 * mu
        elif dRes > 10 * pRes:
            mu = mu / 2

    X = (X + X.t()) / 2
    info['time'] = time.time() - t0
    info['iter'] = n_iter_1

    if eigenvalues:
        info['eigenvalues'] = torch.eig(X)

    X_bin = X > 0.5
    if verbose:
        print(f"Alg terminated. Time = {info['time']}, #Iter = {info['iter']}, Res = ({pRes}, {dRes}), mu = {mu} \n")
    match_mat = transform_closure(X_bin)
    return torch.tensor(match_mat)


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
    return affinity_matrix


from pose_def import Pose
from typing import List
from mv_math_util import Calib


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


def match_multiview_poses(cam_poses: List[List[Pose]], calibs: List[Calib]):
    from mv_math_util import get_fundamental_matrix
    n_cams = len(calibs)
    points_set = []
    dimsGroup = [0]
    cnt = 0
    for poses in cam_poses:
        cnt += len(poses)
        dimsGroup.append(cnt)
        for p in poses:
            points_set.append(p.keypoints)

    points_set = np.array(points_set)
    pairwise_f_mats = calc_pairwise_f_mats(calibs)
    s_mat = geometry_affinity(points_set, pairwise_f_mats, dimsGroup)
    # match_mat = matchSVT(torch.from_numpy(s_mat), dimsGroup)
    match_mat = match_als(s_mat, dimsGroup)

    bin_match = match_mat[:, torch.nonzero(torch.sum(match_mat, dim=0) > 1.9).squeeze()] > 0.9
    bin_match = bin_match.reshape(s_mat.shape[0], -1)
    matched_list = [[] for i in range(bin_match.shape[1])]
    for sub_imgid, row in enumerate(bin_match):
        if row.sum() != 0:
            pid = row.numpy().argmax()
            matched_list[pid].append(sub_imgid)

    outputs = []
    for matches in matched_list:
        cam_p_idxs = []
        for idx in matches:
            cam_offset = 0
            cam_idx = 0
            for cur_cam_idx, offset in enumerate(dimsGroup):
                if offset <= idx:
                    cam_offset = offset
                    cam_idx = cur_cam_idx
                else:
                    break

            p_idx = idx - cam_offset
            cam_p_idxs.append((cam_idx, p_idx))

        if cam_p_idxs:
            outputs.append(cam_p_idxs)

    return outputs


if __name__ == '__main__':
    """
    Unit test, may only work on zjurv2.
    """
    import ipdb
    import pickle
    import os.path as osp
    import sys


    class TempDataset:
        def __init__(self, info_dict, cam_names):
            self.info_dicts = info_dict
            self.cam_names = cam_names

        def __getattr__(self, item):
            if item == 'info_dict':
                return self.info_dicts
            else:
                return self.cam_names


    with open('/home/jiangwen/Multi-Pose/result/0_match.pkl', 'rb') as f:
        d = pickle.load(f)
    test_W = d[1][0].clone()
    test_dimGroup = d[1][1]
    # match_mat = matchSVT(test_W, test_dimGroup, verbose=True)
    match_mat = match_als(np.array(test_W), test_dimGroup, verbose=True)

    ipdb.set_trace()
    bin_match = match_mat[:, torch.nonzero(torch.sum(match_mat, dim=0) > 1.9).squeeze()] > 0.9
    bin_match = bin_match.reshape(test_W.shape[0], -1)
