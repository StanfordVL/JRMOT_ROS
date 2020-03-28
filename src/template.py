#!/home/sibot/anaconda2/bin/python
""" yolo_bbox_to_sort.py
    Subscribe to the Yolo 2 bboxes, and publish the detections with a 2d appearance feature used for reidentification
"""
import time
import rospy
import sys
import torch
import numpy as np
import os
from std_msgs.msg import Int8
import message_filters
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from cv_bridge import CvBridge, CvBridgeError
from aligned_reid_utils import get_image_patches, generate_features, create_appearance_model
from jpda_rospack.msg import detection2d_with_feature_array, detection2d_with_feature

class Appearance_Features:
    def __init__(self):
        self.node_name = "aligned_reid_feature_generator"
        
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)
        apperance_model_ckpt = rospy.get_param('~aligned_reid_model', 'src/jpda_rospack/src/aligned_reid_MOT_weights.pth')
        self.appearance_model = create_appearance_model(apperance_model_ckpt)
        
        self.image_sub = message_filters.Subscriber("/ros_indigosdk_node/stitched_image0", Image, queue_size=2)
        self.yolo_bbox_sub = message_filters.Subscriber("/omni_yolo_bboxes", BoundingBoxes, queue_size=2)
        
        self.time_sync = message_filters.ApproximateTimeSynchronizer([self.yolo_bbox_sub, self.image_sub], 5, 0.1)
        self.time_sync.registerCallback(self.get_2d_feature)
    
        self.cv_bridge = CvBridge()
        self.feature_2d_pub = rospy.Publisher("detection2d_with_feature", detection2d_with_feature_array, queue_size=1)
        self.debug_pub = rospy.Publisher("/test", Int8, queue_size=1)
        rospy.loginfo("Ready.")
        
    def get_2d_feature(self, y1_bboxes, ros_image):
#        rospy.loginfo('Processing Image with AlignedReID')
        start = time.time()
        try:
            input_image = self.cv_bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            print(e)
        input_img = torch.from_numpy(input_image).float()
        input_img = input_img.to('cuda:1')
        input_img = input_img.permute(2, 0, 1)/255
        # Generate 2D image feaures for each bounding box
        detections = []
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
                detections.append([int(xmin), int(ymin), int(xmax), int(ymax), probability, -1, -1])
        features_2d = detection2d_with_feature_array()
        features_2d.header.stamp = y1_bboxes.header.stamp
        features_2d.header.frame_id = 'occam'
        if not detections:
            self.feature_2d_pub.publish(features_2d)
            return
        image_patches = get_image_patches(input_img, detections)
        features = generate_features(self.appearance_model, image_patches)
        
        for (det, feature, i) in zip(detections, features, frame_det_ids):
            det_msg = detection2d_with_feature()
            det_msg.header.stamp = features_2d.header.stamp
            det_msg.x1 = det[0]
            det_msg.y1 = det[1]
            det_msg.x2 = det[2]
            det_msg.y2 = det[3]
            det_msg.feature = feature
            det_msg.valid = True
            det_msg.frame_det_id = i
            features_2d.detection2d_with_features.append(det_msg)
        self.feature_2d_pub.publish(features_2d)
        # rospy.loginfo("Aligned_ReID time: {}".format(time.time() - start))
                
    def cleanup(self):
        print("Shutting down 2D-Appearance node.")
    
def main(args):       
    try:
        Appearance_Features()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down 2D-Appearance node.")

if __name__ == '__main__':
    main(sys.argv)
