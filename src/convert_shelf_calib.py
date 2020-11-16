import numpy as np
from pathlib import Path
import json
import cv2

calib_dir = "/media/F/thesis/data/shelf/calibs"
for js_path in Path(calib_dir).glob('*.json'):
    js_data = json.load(open(js_path))
    K = np.array(js_data["K"]).reshape((3, 3))
    RT = np.array(js_data["RT"]).reshape((3, 4))
    P = K @ RT
    fb = cv2.FileStorage(str(Path(calib_dir) / f'{js_path.stem}.yaml'), cv2.FILE_STORAGE_WRITE)
    fb.write("P", P)
    fb.write("K", K)
    fb.release()
