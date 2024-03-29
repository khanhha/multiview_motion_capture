from typing import List
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.gridspec as gridspec
from collections import defaultdict
import numpy as np
import cv2
import colorsys
from tqdm import tqdm
from pose_def import get_pose_bones_index, Pose

matplotlib.use('Qt5Agg')


def create_unique_color_float(tag: int, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag: int, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


def plot_poses_3d_reprojects(pose_3d_tracks: List[List[Tuple[int, Pose]]],
                             view_img_dir: Path,
                             view_proj_mats: List[np.ndarray],
                             out_vid_path: Path = None,
                             fps=24):
    view_img_paths = defaultdict(list)
    vdirs = sorted([view_dir for view_dir in view_img_dir.glob('*') if view_dir.is_dir()])
    for view_idx, view_dir in enumerate(vdirs):
        view_img_paths[view_idx] = sorted([ipath for ipath in view_dir.glob('*.*')], key=lambda f_: int(f_.stem))

    n_views = len(view_img_paths)
    sample_img = cv2.imread(str(view_img_paths[0][0]))
    n_frames = min([len(view_img_paths[v_i]) for v_i in range(n_views)])
    n_tracklets = len(pose_3d_tracks)

    # frame_idx -> List[[track_id, Pose]]
    frm_tlet_poses: Dict[int, List[Tuple[int, Pose]]] = defaultdict(list)
    for tlet_idx, tlet in enumerate(pose_3d_tracks):
        for frm_idx, p_3d in tlet:
            frm_tlet_poses[frm_idx].append((tlet_idx, p_3d))

    tlet_f_colors = [create_unique_color_float(tlet_idx) for tlet_idx in range(n_tracklets)]
    tlet_u_colors = [(np.array(f_c) * 255.0).astype(np.int) for f_c in tlet_f_colors]
    tlet_u_colors = [(int(u_c[0]), int(u_c[1]), int(u_c[-1])) for u_c in tlet_u_colors]

    # Plot figure with subplots of different sizes
    fig = plt.figure(1)
    # set up subplot grid
    grid_shape = (6, 3)
    gridspec.GridSpec(grid_shape[0], grid_shape[1])

    # large subplot
    ax = plt.subplot2grid(grid_shape, (3, 1), colspan=2, rowspan=3, fig=fig, projection='3d')
    bones_idxs = get_pose_bones_index(pose_3d_tracks[0][0][1].pose_type)
    tlet_bone_lines = []
    for tlet_idx in range(n_tracklets):
        tlet_color = tlet_f_colors[tlet_idx]
        _bones_lines = [ax.plot([0, 0], [0, 0], [0, 0], c=tlet_color)[0] for _ in range(len(bones_idxs))]

        tlet_bone_lines.append(_bones_lines)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # Setting the axes properties
    ax.set_xlim3d([-5.0, 5.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-5.0, 5.0])
    ax.set_zlabel('Z')

    view_img_subplots = [plt.subplot2grid(grid_shape, (0, 0), rowspan=2),
                         plt.subplot2grid(grid_shape, (2, 0), rowspan=2),
                         plt.subplot2grid(grid_shape, (4, 0), rowspan=2),
                         plt.subplot2grid(grid_shape, (0, 1), rowspan=2),
                         plt.subplot2grid(grid_shape, (0, 2), rowspan=2)]
    for img_plt in view_img_subplots:
        img_plt.set_xticklabels([])
        img_plt.set_yticklabels([])

    view_img_axes = [img_plt.imshow(sample_img, animated=True) for img_plt in view_img_subplots]

    bar = tqdm(total=n_frames, desc='output video')

    def _update_pose(_frm_idx):
        bar.update()
        _frm_poses = frm_tlet_poses[_frm_idx]
        _alive_tlets = []
        for _tlet_idx, _pose in _frm_poses:
            _alive_tlets.append(_tlet_idx)
            for _bone, _line in zip(bones_idxs, tlet_bone_lines[_tlet_idx]):
                p0, p1 = _pose.keypoints[_bone[0], :3], _pose.keypoints[_bone[1], :3]
                _line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
                _line.set_3d_properties([p0[2], p1[2]])

        for _v_idx in range(n_views):
            _v_img_path = view_img_paths[_v_idx][_frm_idx]
            _v_img = cv2.imread(str(_v_img_path))
            cam_p = view_proj_mats[_v_idx]
            for _tlet_idx, _pose in _frm_poses:
                _n_kps = len(_pose.keypoints)
                _kps_reproj = cam_p @ np.concatenate([_pose.keypoints, np.ones((_n_kps, 1))], axis=-1).T
                _kps_reproj = (_kps_reproj[:2] / _kps_reproj[2]).T
                _c = tlet_u_colors[_tlet_idx][::-1]
                for _bone in bones_idxs:
                    p0, p1 = _kps_reproj[_bone[0], :2], _kps_reproj[_bone[1], :2]
                    x0, y0 = p0.astype(np.int)
                    x1, y1 = p1.astype(np.int)
                    cv2.line(_v_img, (x0, y0), (x1, y1), _c, thickness=2)

            view_img_axes[_v_idx].set_array(_v_img[:, :, ::-1])

        all_lines = []
        for _tlet_idx in range(n_tracklets):
            if _tlet_idx not in _alive_tlets:
                for _line in tlet_bone_lines[_tlet_idx]:
                    _line.set_data([0, 0], [0, 0])
                    _line.set_3d_properties([0, 0])
                    all_lines.append(_line)

        return all_lines

    # fit subplots and save fig
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    fig.set_size_inches(w=20, h=20)
    # Creating the Animation object
    # Set up formatting for the movie files
    anim = animation.FuncAnimation(fig, _update_pose, n_frames, interval=50, blit=True)
    if out_vid_path is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(str(out_vid_path), writer=writer)
    else:
        plt.show()

    bar.close()


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
