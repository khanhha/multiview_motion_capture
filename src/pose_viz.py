from typing import List
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import cv2
from pose_def import get_pose_bones_index, Pose

matplotlib.use('Qt5Agg')


def plot_poses_3d(poses_3d: List[Pose], fps=24):
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    bones_idxs = get_pose_bones_index(poses_3d[0].pose_type)
    bones_lines = [ax.plot([0, 0], [0, 0], [0, 0])[0] for _ in range(len(bones_idxs))]

    def _update_pose(frm_idx):
        for bone, line in zip(bones_idxs, bones_lines):
            p0, p1 = poses_3d[frm_idx].keypoints[bone[0], :3], poses_3d[frm_idx].keypoints[bone[1], :3]
            line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
            line.set_3d_properties([p0[2], p1[2]])

    # Setting the axes properties
    ax.set_xlim3d([-5.0, 5.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-5.0, 5.0])
    ax.set_zlabel('Z')

    # Creating the Animation object
    line_anim = animation.FuncAnimation(fig, _update_pose, len(poses_3d), interval=50, blit=False)
    plt.show()


def pose_to_bb(pose: Pose, min_valid_kps=0.1):
    valid_kps = pose.keypoints[pose.keypoints_score.flatten() > min_valid_kps, :]
    bmin, bmax = np.min(valid_kps, axis=0), np.max(valid_kps, axis=0)
    return np.concatenate([bmin, bmax])


def draw_pose(img: np.ndarray, pose: Pose, crop: bool, draw_bb=True):
    for i in range(len(pose.keypoints)):
        x, y = pose.keypoints[i, :2].astype(np.int)
        cv2.circle(img, (x, y), 2, (255, 0, 0), thickness=2, lineType=cv2.FILLED)

    if draw_bb and pose.box is not None:
        x0, y0, x1, y1 = pose.box.astype(np.int)
        cv2.rectangle(img, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=2)

    if crop:
        bb = pose_to_bb(pose)
        x1, y1, x2, y2 = bb.astype(np.int)
        bb_img = img[y1:y2, x1:x2]
        return bb_img
    else:
        return img


def draw_poses_concat(poses: List[Pose], imgs: List[np.ndarray],
                      pose_texts: Optional[List[str]],
                      top_text=Optional[str], crop_height=256, draw_kps=True):
    all_crops = []
    for idx, (pose, img_) in enumerate(zip(poses, imgs)):
        img = img_.copy()
        x1, y1, x2, y2 = pose_to_bb(pose).astype(np.int)
        is_valid_box = y2 > y1 + 5 and x2 > x1 + 5
        if draw_kps:
            n_kps = len(pose.keypoints)
            for kps_idx in range(n_kps):
                x, y = pose.keypoints[kps_idx, :2].astype(np.int)
                cv2.circle(img, (x, y), 2, color=(0, 0, 255), thickness=2, lineType=cv2.FILLED)

        if is_valid_box:
            crop = img[y1:y2, x1:x2]
        else:
            crop = img
        c_h, c_w = crop.shape[:2]
        new_h = crop_height
        new_w = int((c_w / c_h) * new_h)
        crop = cv2.resize(crop, dsize=(new_w, new_h))
        if pose_texts is not None:
            cv2.putText(crop, f'{pose_texts[idx]}', (int(0.05 * new_w), int(0.4 * crop_height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), thickness=1)
        all_crops.append(crop)

    # view_id increasing from left to right
    if all_crops:
        viz = np.concatenate(all_crops, axis=1)
        if top_text is not None:
            cv2.putText(viz, top_text, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
        return viz
    else:
        return None


def draw_pose_epiplar_lines(img1, img2, pose0, pose1, f_mat):
    pts1 = pose0.keypoints
    pts2 = pose1.keypoints
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape((-1, 1, 2)), 2, f_mat)
    lines1 = lines1.reshape(-1, 3)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape((-1, 1, 2)), 1, f_mat)
    lines2 = lines2.reshape(-1, 3)

    img1, img2 = draw_epipolar_lines(img1, img2, lines1, pts1, pts2)

    img1, img2 = draw_epipolar_lines(img2, img1, lines2, pts2, pts1)

    return img1, img2


def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
    """
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    """
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist()[:])
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1.astype(np.int32)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2.astype(np.int32)), 5, color, -1)
    return img1, img2
