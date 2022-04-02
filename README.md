
# Cross view tracking and inverse kinematics
This is a prototype implementation in python of the paper "[Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS
](https://arxiv.org/abs/2003.03972)".

![image alt ><](https://user-images.githubusercontent.com/2276264/161385537-60e30983-95c8-4123-af8f-de25821eae68.png)

In addition to the original ideas from the paper, the implementation also includes
- a temporal inverse kinematics solver that transform 3d triangulated points into joint angles 
- a temporal bone length optimization. 

Instead of cross-view tracking using graph partition as described in the paper, this implementation
use a greedy approach to associate 2d poses across views.

# Install  
- install [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). Note that if you just want to visualize the captured
animation or do some test with the code,  you can ignore this step. I already re-generated some intermediate data
and the final captured animation of the Shelf dataset under the ./data folder.

- create anaconda environment
```bash
conda env create -n motion python=3.8.2
conda activate motion
pip install -r requirements.txt
```

# Usage
This is the complete instruction to run the whole pipeline.

- __run openpose__: the following bash script will call openpose to extract 2d keypoints from input videos and output json result
to the output folder [the second argument]
```
bash ./run_openpose.sh ./data/shelf/videos ./data/shelf/kps_opn
```

- __generate the test data__: prepare some convenient data for running the pipeline. In principle, the pipeline
can be run in real-time, but for the sake of debugging, it's more convenient to export some pre-generated data.

```
python ./motion_capture.py --mode prepare --opn_kps_dir ./data/shelf/kps_opn --calib_dir ./data/shelf/calibs --out_data_dir ./data/shelf/dframes/
```

- __cross-view tracking and inverse kinematics__
```
python ./motion_capture.py --model run --video_dir ./data/shelf/videos --data_dir ./data/shelf/dframes --output_dir ./data/shelf/tracklets/
```

- __visualize__ the captured animation
```
python ./motion_capture.py --tlet_path ./data/shelf/tracklets/tracklets.pkl
```

#TODO
- debug frame 131. black guy is failed
