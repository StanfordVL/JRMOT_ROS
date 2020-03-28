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
import pdb
import time
import os
import cv2
from std_msgs.msg import Int8
import message_filters
from sensor_msgs.msg import PointCloud2, Image
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from featurepointnet_model_util import generate_detections_3d, convert_depth_features
from featurepointnet_model import create_depth_model
from calibration import OmniCalibration
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, Vector3
from std_msgs.msg import ColorRGBA
from jpda_rospack.msg import detection3d_with_feature_array, detection3d_with_feature

class Detector_3d:
    def __init__(self):
        self.node_name = "fpointnet_detector_plus_feature"
        
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)
        fpointnet_config = \
            rospy.get_param('~fpointnet_config',
                            '~/jr2_catkin_workspace/src/jpda_rospack/src/fpointnet_jrdb/model.ckpt')
        calibration_folder = rospy.get_param('~calib_3d', 'src/jpda_rospack/calib/')
        self.depth_model = create_depth_model('FPointNet', fpointnet_config)
        self.calib = OmniCalibration(calibration_folder)
        self.velodyne_sub_upper = \
            message_filters.Subscriber("/upper_velodyne/velodyne_points", PointCloud2, queue_size=2)
        self.velodyne_sub_lower = \
            message_filters.Subscriber("/lower_velodyne/velodyne_points", PointCloud2, queue_size=2)
        self.yolo_bbox_sub = \
            message_filters.Subscriber("/omni_yolo_bboxes", BoundingBoxes, queue_size=2)
        
        self.time_sync = \
            message_filters.ApproximateTimeSynchronizer([self.yolo_bbox_sub,
                                                         self.velodyne_sub_upper,
                                                         self.velodyne_sub_lower], 5, 0.06)
        self.time_sync.registerCallback(self.get_3d_feature)
    
        self.feature_3d_pub = rospy.Publisher("detection3d_with_feature", detection3d_with_feature_array, queue_size=10)
        self.pc_transform_pub = rospy.Publisher("/transformed_pointcloud", PointCloud2, queue_size=10)
        self.pc_pub = rospy.Publisher("/frustum", PointCloud2, queue_size=10)
        self.debug_pub = rospy.Publisher("/test", Int8, queue_size=1)
        self.marker_box_pub = rospy.Publisher("/3d_detection_markers", MarkerArray, queue_size=10)
        rospy.loginfo("3D detector ready.")
        
    def get_3d_feature(self, y1_bboxes, pointcloud_upper, pointcloud_lower):
        start = time.time()
        #rospy.loginfo('Processing Pointcloud with FPointNet')
        # Assumed that pointclouds have 64 bit floats!
        pc_upper = ros_numpy.numpify(pointcloud_upper).astype({'names':['x','y','z','intensity','ring'], 'formats':['f4','f4','f4','f4','f4'], 'offsets':[0,4,8,16,20], 'itemsize':32})
        pc_lower = ros_numpy.numpify(pointcloud_lower).astype({'names':['x','y','z','intensity','ring'], 'formats':['f4','f4','f4','f4','f4'], 'offsets':[0,4,8,16,20], 'itemsize':32})
        pc_upper = torch.from_numpy(pc_upper.view(np.float32).reshape(pc_upper.shape + (-1,)))[:, [0,1,2,4]]
        pc_lower = torch.from_numpy(pc_lower.view(np.float32).reshape(pc_lower.shape + (-1,)))[:, [0,1,2,4]]
        # move onto gpu if available
        try:
            pc_upper = pc_upper.cuda()
            pc_lower = pc_lower.cuda()
        except:
            pass
        # translate and rotate into camera frame using calib object
        # in message pointcloud has x pointing forward, y pointing to the left and z pointing upward
        # need to transform this such that x is pointing to the right, y pointing downwards, z pointing forward
        # also done inside calib
        pc_upper = self.calib.move_lidar_to_camera_frame(pc_upper, upper=True)
        pc_lower = self.calib.move_lidar_to_camera_frame(pc_lower, upper=False)
        pc = torch.cat([pc_upper, pc_lower], dim = 0)
        pc[:, 3] = 1
        # pc = pc.cpu().numpy()
        # self.publish_pointcloud_from_array(pc, self.pc_transform_pub, header = pointcloud_upper.header)
        # idx = torch.randperm(pc.shape[0]).cuda()
        # pc = pc[idx]
        detections_2d = []
        frame_det_ids = []
        count = 0
        for y1_bbox in y1_bboxes.bounding_boxes:
            if y1_bbox.Class == 'person':
                xmin = y1_bbox.xmin
                xmax = y1_bbox.xmax
                ymin = y1_bbox.ymin
                ymax = y1_bbox.ymax
                probability = y1_bbox.probability
                frame_det_ids.append(count)
                count += 1
                detections_2d.append([xmin, ymin, xmax, ymax, probability, -1, -1])
        features_3d = detection3d_with_feature_array()
        features_3d.header.stamp = y1_bboxes.header.stamp
        features_3d.header.frame_id = 'occam'
        boxes_3d_markers = MarkerArray()
        if not detections_2d:
            self.marker_box_pub.publish(boxes_3d_markers)
            self.feature_3d_pub.publish(features_3d)
            return
        boxes_3d, valid_3d, rot_angles, _, depth_features, frustums = \
            generate_detections_3d(self.depth_model, detections_2d, pc,
                                   self.calib, (3, 480, 3760), omni=True,
                                   peds=True)
        depth_features = convert_depth_features(depth_features, valid_3d)

        for box, feature, i in zip(boxes_3d, depth_features, frame_det_ids):
            #frustum = frustums[i]
            #frustum[:, [0,2]] = np.squeeze(np.matmul(
            #                 np.array([[np.cos(rot_angles[i]), np.sin(rot_angles[i])], 
            #                 [-np.sin(rot_angles[i]), np.cos(rot_angles[i])]]), 
            #                 np.expand_dims(frustum[:, [0,2]], 2)), 2)
            # frustum[:, 3] = np.amax(logits[i], axis = 1)
            #self.publish_pointcloud_from_array(frustum, self.pc_pub, header = pointcloud_upper.header)
            det_msg = detection3d_with_feature()
            det_msg.header.frame_id = 'occam'
            det_msg.header.stamp = features_3d.header.stamp
            det_msg.valid = True if valid_3d[i] != -1 else False
            det_msg.frame_det_id = i
            if det_msg.valid:
                det_msg.x = box[0]
                det_msg.y = box[1]
                det_msg.z = box[2]
                det_msg.l = box[3]
                det_msg.h = box[4]
                det_msg.w = box[5]
                det_msg.theta = box[6]
                det_msg.feature = feature
                features_3d.detection3d_with_features.append(det_msg)
                pose_msg = Pose()
                marker_msg = Marker()
                marker_msg.header.stamp = pointcloud_lower.header.stamp
                marker_msg.header.frame_id = 'occam'
                marker_msg.action = 0
                marker_msg.id = i
                marker_msg.lifetime = rospy.Duration(0.2)
                marker_msg.type = 1
                marker_msg.scale = Vector3(box[3], box[4], box[5])
                pose_msg.position.x = det_msg.x
                pose_msg.position.y = det_msg.y - det_msg.h/2
                pose_msg.position.z = det_msg.z
                marker_msg.pose = pose_msg
                marker_msg.color = ColorRGBA(g=1, a =0.5)
                boxes_3d_markers.markers.append(marker_msg)
            else:
                det_msg.y = -1
                det_msg.x = -1
                det_msg.z = -1
                det_msg.l = -1
                det_msg.w = -1
                det_msg.h = -1
                det_msg.theta = -1
                det_msg.feature = [-1]
            features_3d.detection3d_with_features.append(det_msg)

        
        self.marker_box_pub.publish(boxes_3d_markers)
        
        self.feature_3d_pub.publish(features_3d)
        
        # rospy.loginfo("3D detector time: {}".format(time.time() - start))

    def publish_pointcloud_from_array(self, pointcloud, publisher, frame = 'occam', header = None):
        list_pc = [tuple(j) for j in pointcloud]
        pc_output_msg = np.array(list_pc, dtype = [('x', 'f4'),('y', 'f4'),('z', 'f4'),('intensity', 'f4')])
        pc_msg = ros_numpy.msgify(PointCloud2, pc_output_msg)
        if header is not None:
            pc_msg.header.stamp = header.stamp
        pc_msg.header.frame_id = 'occam'
        publisher.publish(pc_msg)

    def cleanup(self):
        print("Shutting down 3D-Detection node.")
    
def main(args):       
    try:
        Detector_3d()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down 3D-Detection node.")

if __name__ == '__main__':
    main(sys.argv)
