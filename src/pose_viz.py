from typing import List
from pathlib import Path
from typing import List

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


def draw_pose(img: np.ndarray, pose: Pose, crop: bool):
    for i in range(len(pose.keypoints)):
        x, y = pose.keypoints[i, :2].astype(np.int)
        cv2.circle(img, (x, y), 2, (255, 0, 0), thickness=2, lineType=cv2.FILLED)

    if crop:
        bb = pose_to_bb(pose)
        x1, y1, x2, y2 = bb.astype(np.int)
        bb_img = img[y1:y2, x1:x2]
        return bb_img
    else:
        return img


def draw_poses_concat(poses: List[Pose], imgs: List[np.ndarray],
                      view_idxs=None, frm_idx=None, crop_height=256):
    all_crops = []
    for idx, (pose, img) in enumerate(zip(poses, imgs)):
        x1, y1, x2, y2 = pose_to_bb(pose).astype(np.int)
        if y2 > y1 + 5 and x2 > x1 + 5:
            crop = img[y1:y2, x1:x2]
            c_h, c_w = crop.shape[:2]
            new_h = crop_height
            new_w = int((c_w / c_h) * new_h)
            crop = cv2.resize(crop, dsize=(new_w, new_h))
            if view_idxs is not None:
                cv2.putText(crop, f'{view_idxs[idx]}', (int(0.5 * new_w), 128),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
            all_crops.append(crop)

    # view_id increasing from left to right
    if all_crops:
        viz = np.concatenate(all_crops, axis=1)
        if frm_idx is not None:
            cv2.putText(viz, f'{frm_idx}', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
        return viz
    else:
        return None
