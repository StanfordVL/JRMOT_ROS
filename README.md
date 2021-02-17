# JRMOT ROS package

The repository contains the code for the work "JRMOT: A Real-Time 3D Multi-Object Tracker and a New Large-Scale Dataset".

Note that due to the global pandemic, this repository is still a work in progress. Updates will be made as soon as possible.

## Introduction

JRMOT is a 3D multi object tracking system that:
- Is real-time
- Is online
- Fuses 2D and 3D information
- Achieves State of the Art performance on KITTI

We also release JRDB:
- A dataset with over 2 million annotated boxes and 3500 time consistent trajectories in 2D and 3D
- Captured in social, human-centric settings
- Captured by our social mobile-manipulator JackRabbot
- Contains 360 degree cylindrical images, stereo camera images, 3D pointclouds and more sensing modalties

All information, including download links for JRDB can be found [here](https://jrdb.stanford.edu).

## JRMOT
![system overview](https://github.com/StanfordVL/JRMOT_ROS/blob/master/assets/framework.png)

- Our system is built on top of state of the art 2D and 3D detectors (mask-RCNN and F-PointNet respectively). These detections are associated with predicted track locations at every time step. 
- Association is done via a novel feature fusion, as well as a cost selection procedure, followed by Kalman state gating and JPDA. 
- Given the JPDA output, we use both 2D and 3D detections in a novel multi-modal Kalman filter to update the track locations.


## Using the code

There are 3 nodes forming parts of the ROS package:
+ 3d_detector.py: Runs F-PointNet, which performs 3D detection and 3D feature extraction
+ template.py: Runs Aligned-Re-ID, which performs 2D feature extraction
+ tracker_3d_node.py: Performs tracking while taking both 2D detections + features and 3D detections + features as input

The launch file in the folder "launch" launches all 3 nodes.

## Dependencies

The following are dependencies of the code:

+ 2D detector: The 2D detector is not included in this package. To interface with your own 2D detector, please modify the file template.py to subscribe to the correct topic, and also to handle the conversion from ROS message to numpy array.
+ Spencer People Tracking messages: The final tracker output is in a Spencer People Tracking message. Please install this package and include these message types.
+ Various python packages: These can be found in [requirements.txt](./requirements.txt).. Please install all dependencies prior to running the code (including CUDA and cuDNN. Additionally, this code requires a solver called Gurobi. Instructions to install gurobipy can be found [here](https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html).
+ Weight files: The trained weights, (trained on JRDB) for FPointNet and Aligne-ReID can be found [here](https://drive.google.com/open?id=1YQinMPVWEI44KezS9inXe0mvVnm4aL3s).

## Citation

If you find this work useful, please cite:
```
@INPROCEEDINGS{shenoi2020jrmot,
  author={A. {Shenoi} and M. {Patel} and J. {Gwak} and P. {Goebel} and A. {Sadeghian} and H. {Rezatofighi} and R. {Mart\’in-Mart\’in} and S. {Savarese}},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={JRMOT: A Real-Time 3D Multi-Object Tracker and a New Large-Scale Dataset}, 
  year={2020},
  volume={},
  number={},
  pages={10335-10342},
  doi={10.1109/IROS45743.2020.9341635}}
```

If you utilise our dataset, please also cite:

```
@article{martin2019jrdb,
  title={JRDB: A dataset and benchmark of egocentric visual perception for navigation in human environments},
  author={Mart{\'\i}n-Mart{\'\i}n, Roberto and Patel, Mihir and Rezatofighi, Hamid and Shenoi, Abhijeet and Gwak, JunYoung and Frankel, Eric and Sadeghian, Amir and Savarese, Silvio},
  journal={arXiv preprint arXiv:1910.11792},
  year={2019}
}
```
## 
