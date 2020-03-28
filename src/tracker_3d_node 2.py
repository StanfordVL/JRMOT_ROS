#!/home/sibot/anaconda2/bin/python
""" yolo_bbox_to_sort.py
    Subscribe to the Yolo 2 bboxes, and publish the detections with a 2d appearance feature used for reidentification
"""
import time
import rospy
import ros_numpy
import sys
import numpy as np
import torch
import os
import message_filters
from featurepointnet_model_util import generate_detections_3d, \
    convert_depth_features
from featurepointnet_model import create_depth_model
from calibration import OmniCalibration
from jpda_rospack.msg import detection3d_with_feature_array, \
    detection3d_with_feature, detection2d_with_feature_array
from tracking_utils import convert_detections, combine_features
from combination_model import CombiNet
from tracker_3d import Tracker_3d
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Int8
from geometry_msgs.msg import Pose, PoseWithCovariance
from spencer_tracking_msgs.msg import TrackedPerson, TrackedPersons

import pdb


class Tracker_3D_node:
    def __init__(self):
        self.node_name = "tracker_3d"
        
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.depth_weight = float(rospy.get_param('~combination_depth_weight', 1))
        calibration_folder = rospy.get_param('~calib_3d', 'src/jpda_rospack/calib/')
        calib = OmniCalibration(calibration_folder)
        self.tracker = Tracker_3d(max_age=25, n_init=3,
                                  JPDA=True, m_best_sol=10, assn_thresh=0.6,
                                  matching_strategy='hungarian',
                                  cuda=True, calib=calib, omni=True,
                                  kf_vel_params=(0.08, 0.03, 0.01, 0.03,
                                                 1.2, 3.9, 0.8, 1.6),
                                  dummy_node_cost_iou=0.9, dummy_node_cost_app=6,
                                  nn_budget=3, dummy_node_cost_iou_2d=0.5)

        combination_model_path = rospy.get_param('~combination_model_path', False)
        if combination_model_path:
            self.combination_model = CombiNet()
            checkpoint = torch.load(combination_model_path)
            self.combination_model.load_state_dict(checkpoint['state_dict'])
            try:
                combination_model.cuda()
            except:
                pass
            self.combination_model.eval()
        else:
            self.combination_model = None
        
        self.detection_2d_sub = \
            message_filters.Subscriber("detection2d_with_feature",
                                       detection2d_with_feature_array,
                                       queue_size=5)
        self.detection_3d_sub = \
            message_filters.Subscriber("detection3d_with_feature",
                                       detection3d_with_feature_array,
                                       queue_size=5)
        
        # self.detection_2d_sub.registerCallback(self.find_time_diff_2d)
        # self.detection_3d_sub.registerCallback(self.find_time_diff_3d)
        # self.last_seen_2d = 0
        # self.last_seen_3d = 0
        self.time_sync = \
            message_filters.TimeSynchronizer([self.detection_2d_sub,
                                                         self.detection_3d_sub],
                                                        5)
        self.time_sync.registerCallback(self.do_3d_tracking)
    
        self.tracker_output_pub = rospy.Publisher("/jpda_output", TrackedPersons,
                                                  queue_size=30)
    
        self.debug_pub = rospy.Publisher("/test", Int8, queue_size=1)
        rospy.loginfo("Ready.")
        
    def do_3d_tracking(self, detections_2d, detections_3d):
        start = time.time()
        #rospy.loginfo("Tracking frame")
        # convert_detections
        boxes_2d = []
        boxes_3d = []
        valid_3d = []
        features_2d = []
        features_3d = []
        dets_2d = sorted(detections_2d.detection2d_with_features, key=lambda x:x.frame_det_id)
        dets_3d = sorted(detections_3d.detection3d_with_features, key=lambda x:x.frame_det_id)
        i, j = 0, 0
        while i < len(dets_2d) and j < len(dets_3d):
            det_2d = dets_2d[i]
            det_3d = dets_3d[j]
            if det_2d.frame_det_id == det_3d.frame_det_id:
                i += 1
                j += 1
                valid_3d.append(det_3d.valid)
                boxes_2d.append(np.array([det_2d.x1, det_2d.y1, det_2d.x2, det_2d.y2, 1, -1, -1]))
                features_2d.append(torch.Tensor(det_2d.feature).to('cuda:0'))
                if det_3d.valid:
                    boxes_3d.append(np.array([det_3d.x, det_3d.y, det_3d.z, det_3d.l, det_3d.h, det_3d.w, det_3d.theta]))
                    features_3d.append(torch.Tensor(det_3d.feature).to('cuda:0'))
                else:
                    boxes_3d.append(None)
                    features_3d.append(None)
            elif det_2d.frame_det_id < det_3d.frame_det_id:
                i += 1
            else:
                j += 1
        
        if not boxes_3d:
            boxes_3d = None
        features_3d, features_2d = combine_features(features_2d, features_3d,
                                                    valid_3d, self.combination_model,
                                                    depth_weight=self.depth_weight)
        detections = convert_detections(boxes_2d, features_3d, features_2d, boxes_3d)
        self.tracker.predict()
        self.tracker.update(None, detections)
        tracked_array = TrackedPersons()
        tracked_array.header.stamp = detections_3d.header.stamp
        tracked_array.header.frame_id = 'occam'

        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue
            #print('Confirmed track!')
            pose_msg = Pose()
            tracked_person_msg = TrackedPerson()
            tracked_person_msg.header.stamp = detections_3d.header.stamp
            tracked_person_msg.header.frame_id = 'occam'
            tracked_person_msg.track_id = track.track_id
            if track.time_since_update < 2:
                tracked_person_msg.is_matched = True
            else:
                tracked_person_msg.is_matched = False
            bbox = track.to_tlwh3d()
            covariance = track.get_cov().reshape(-1).tolist()
            pose_msg.position.x = bbox[0]
            pose_msg.position.y = bbox[1] - bbox[4]/2
            pose_msg.position.z = bbox[2]
            pose_msg = PoseWithCovariance(pose=pose_msg, covariance=covariance)
            tracked_person_msg.pose = pose_msg
            tracked_array.tracks.append(tracked_person_msg)

        self.tracker_output_pub.publish(tracked_array)

        #rospy.loginfo("tracker time: {}".format(time.time() - start))

    def find_time_diff_2d(self, a):
        print(a.header.stamp - self.last_seen_3d)
        self.last_seen_2d = a.header.stamp

    def find_time_diff_3d(self, a):
        print(a.header.stamp - self.last_seen_2d)
        self.last_seen_3d = a.header.stamp

    def cleanup(self):
        print("Shutting down 3D tracking node.")
        del self.combination_model
        del self.tracker
        del self.detection_2d_sub
        del self.detection_3d_sub
        del self.time_sync
        del self.tracker_output_pub
    
def main(args):       
    try:
        Tracker_3D_node()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down 3D tracking node.")

if __name__ == '__main__':
    main(sys.argv)
