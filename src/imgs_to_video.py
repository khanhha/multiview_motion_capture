import imageio
from pathlib import Path
import cv2

img_dir = '/media/F/thesis/data/debug/tmp_1'
out_vid = '/media/F/thesis/data/debug/track0.mp4'
h, w = 365, 512

writer = imageio.get_writer(out_vid, fps=5)

img_paths = sorted([ipath for ipath in Path(img_dir).glob('*.jpg')], key=lambda f_: int(f_.stem.split('_')[0]))
for ipath in img_paths:
    img = imageio.imread(ipath)
    img = cv2.resize(img, dsize=(w, h))
    writer.append_data(img)
writer.close()
