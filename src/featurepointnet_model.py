import os, pdb
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import configparser

import featurepointnet_tf_util as tf_util
import featurepointnet_model_util as model_util
from calibration import Calibration, OmniCalibration

batch_size = 6 #TODO: Update if needed?

class FPointNet():
    def __init__(self, config_path):
        parser = configparser.SafeConfigParser()
        parser.read(config_path)
        self.num_point = parser.getint('general', 'num_point')
        self.model_path = parser.get('general', 'model_path')

        with tf.device('/gpu:'+str('0')):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = model_util.placeholder_inputs(batch_size, self.num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points, depth_feature = self.get_model(pointclouds_pl, one_hot_vec_pl, is_training_pl)
            self.object_pointcloud = tf.placeholder(tf.float32, shape=(None, None, 3))
            #depth_feature = self.get_depth_feature_op(is_training_pl)
            loss = model_util.get_loss(labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl, size_class_label_pl, size_residual_label_pl, end_points)
            self.saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        #Initialize variables
        self.sess.run(tf.global_variables_initializer())
        # Restore variables from disk.
        self.saver.restore(self.sess, self.model_path)
        self.ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'center': end_points['center'],
               'end_points': end_points,
               'depth_feature':depth_feature,
               'loss': loss}

    # @profile
    def __call__(self, input_point_cloud, rot_angle, peds=False):
        '''
        one_hot_vec = np.zeros((batch_size, 3))
        feed_dict = {self.pointclouds_pl: input_point_cloud,
                     self.one_hot_vec_pl: one_hot_vec,
                     self.is_training_pl: False}
        features = self.sess.run(self.feature,feed_dict=feed_dict)
        return features '''

        ''' Run inference for frustum pointnets in batch mode '''
        
        one_hot_vec = np.zeros((batch_size,3))
        if peds:
            one_hot_vec[:, 1] = 1
        num_batches = input_point_cloud.shape[0]//batch_size + 1
        num_inputs = input_point_cloud.shape[0]
        if input_point_cloud.shape[0]%batch_size !=0:
            input_point_cloud = np.vstack([input_point_cloud, np.zeros((batch_size - input_point_cloud.shape[0]%batch_size, self.num_point, 4))])
        else:
            num_batches -= 1
        logits = np.zeros((input_point_cloud.shape[0], input_point_cloud.shape[1], 2))
        centers = np.zeros((input_point_cloud.shape[0], 3))
        heading_logits = np.zeros((input_point_cloud.shape[0], model_util.NUM_HEADING_BIN))
        heading_residuals = np.zeros((input_point_cloud.shape[0], model_util.NUM_HEADING_BIN))
        size_logits = np.zeros((input_point_cloud.shape[0], model_util.NUM_SIZE_CLUSTER))
        size_residuals = np.zeros((input_point_cloud.shape[0], model_util.NUM_SIZE_CLUSTER, 3))
        scores = np.zeros((input_point_cloud.shape[0],)) # 3D box score 
        features = np.zeros((input_point_cloud.shape[0], 512))
        
        for i in range(num_batches):    
            ep = self.ops['end_points'] 
            feed_dict = {\
                self.ops['pointclouds_pl']: input_point_cloud[i*batch_size: (i+1)*batch_size],
                self.ops['one_hot_vec_pl']: one_hot_vec,
                self.ops['is_training_pl']: False}

            batch_centers, \
            batch_heading_scores, batch_heading_residuals, \
            batch_size_scores, batch_size_residuals, batch_features = \
                self.sess.run([self.ops['center'],
                    ep['heading_scores'], ep['heading_residuals'],
                    ep['size_scores'], ep['size_residuals'], self.ops['depth_feature']],
                    feed_dict=feed_dict)

            # logits[i*batch_size: (i+1)*batch_size] = batch_logits
            centers[i*batch_size: (i+1)*batch_size] = batch_centers
            heading_logits[i*batch_size: (i+1)*batch_size] = batch_heading_scores
            heading_residuals[i*batch_size: (i+1)*batch_size] = batch_heading_residuals
            size_logits[i*batch_size: (i+1)*batch_size] = batch_size_scores
            size_residuals[i*batch_size: (i+1)*batch_size] = batch_size_residuals
            features[i*batch_size: (i+1)*batch_size] = batch_features[:,0,:]
        heading_cls = np.argmax(heading_logits, 1) # B
        size_cls = np.argmax(size_logits, 1) # B
        heading_res = np.vstack([heading_residuals[i, heading_cls[i]] for i in range(heading_cls.shape[0])])
        size_res = np.vstack([size_residuals[i, size_cls[i], :] for i in range(size_cls.shape[0])])

        #TODO: Make this accept batches if wanted
        boxes = []
        for i in range(num_inputs):
            box = np.array(model_util.from_prediction_to_label_format(centers[i], heading_cls[i], heading_res[i], size_cls[i], size_res[i], rot_angle[i]))
            box[6] = np.squeeze(box[6])
            swp = box[5]
            box[5] = box[4]
            box[4] = swp
            boxes.append(box)       
        boxes = np.vstack(boxes)
        return boxes, scores[:num_inputs], features[:num_inputs]


    def get_instance_seg_v1_net(self, point_cloud, one_hot_vec, is_training, bn_decay, end_points):
        ''' 3D instance segmentation PointNet v1 network.
        Input:
            point_cloud: TF tensor in shape (B,N,4)
                frustum point clouds with XYZ and intensity in point channels
                XYZs are in frustum coordinate
            one_hot_vec: TF tensor in shape (B,3)
                length-3 vectors indicating predicted object type
            is_training: TF boolean scalar
            bn_decay: TF float scalar
            end_points: dict
        Output:
            logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
            end_points: dict
        '''
        num_point = point_cloud.get_shape()[1].value

        net = tf.expand_dims(point_cloud, 2)

        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        point_feat = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(point_feat, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        global_feat = tf_util.max_pool2d(net, [num_point,1],
                                         padding='VALID', scope='maxpool')

        global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)
        global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
        concat_feat = tf.concat(axis=3, values=[point_feat, global_feat_expand])

        net = tf_util.conv2d(concat_feat, 512, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv6', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv7', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv8', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv9', bn_decay=bn_decay)
        net = tf_util.dropout(net, is_training, 'dp1', keep_prob=0.5)

        logits = tf_util.conv2d(net, 2, [1,1],
                             padding='VALID', stride=[1,1], activation_fn=None,
                             scope='conv10')
        logits = tf.squeeze(logits, [2]) # BxNxC
        return logits, end_points
     
    def get_3d_box_estimation_v1_net(self, object_point_cloud, one_hot_vec,is_training, bn_decay, end_points):
        ''' 3D Box Estimation PointNet v1 network.
        Input:
            object_point_cloud: TF tensor in shape (B,M,C)
                point clouds in object coordinate
            one_hot_vec: TF tensor in shape (B,3)
                length-3 vectors indicating predicted object type
        Output:
            output: TF tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
                including box centers, heading bin class scores and residuals,
                and size cluster scores and residuals
        ''' 
        num_point = object_point_cloud.get_shape()[1].value
        net = tf.expand_dims(object_point_cloud, 2)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 512, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg4', bn_decay=bn_decay)

        features = tf.reduce_max(net, axis = 1)

        net = tf_util.max_pool2d(net, [num_point,1],
            padding='VALID', scope='maxpool2')
        net = tf.squeeze(net, axis=[1,2])
        net = tf.concat([net, one_hot_vec], axis=1)
        net = tf_util.fully_connected(net, 512, scope='fc1', bn=True,
            is_training=is_training, bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, scope='fc2', bn=True,
            is_training=is_training, bn_decay=bn_decay)

        # The first 3 numbers: box center coordinates (cx,cy,cz),
        # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
        # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
        output = tf_util.fully_connected(net,
            3+model_util.NUM_HEADING_BIN*2+model_util.NUM_SIZE_CLUSTER*4, activation_fn=None, scope='fc3')
        return output, end_points, features

    def get_model(self, point_cloud, one_hot_vec, is_training, bn_decay=None):
        ''' Frustum PointNets model. The model predict 3D object masks and
        amodel bounding boxes for objects in frustum point clouds.
        Input:
            point_cloud: TF tensor in shape (B,N,4)
                frustum point clouds with XYZ and intensity in point channels
                XYZs are in frustum coordinate
            one_hot_vec: TF tensor in shape (B,3)
                length-3 vectors indicating predicted object type
            is_training: TF boolean scalar
            bn_decay: TF float scalar
        Output:
            end_points: dict (map from name strings to TF tensors)
        '''
        end_points = {}
        
        # 3D Instance Segmentation PointNet
        logits, end_points = self.get_instance_seg_v1_net(\
            point_cloud, one_hot_vec,
            is_training, bn_decay, end_points)
        end_points['mask_logits'] = logits

        # Masking
        # select masked points and translate to masked points' centroid
        object_point_cloud_xyz, mask_xyz_mean, end_points = \
            model_util.point_cloud_masking(point_cloud, logits, end_points)

        # T-Net and coordinate translation
        center_delta, end_points = model_util.get_center_regression_net(\
            object_point_cloud_xyz, one_hot_vec,
            is_training, bn_decay, end_points)
        stage1_center = center_delta + mask_xyz_mean # Bx3
        end_points['stage1_center'] = stage1_center
        # Get object point cloud in object coordinate
        object_point_cloud_xyz_new = \
            object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

        # Amodel Box Estimation PointNet
        output, end_points, features = self.get_3d_box_estimation_v1_net(\
            object_point_cloud_xyz_new, one_hot_vec,
            is_training, bn_decay, end_points)

        # Parse output to 3D box parameters
        end_points = model_util.parse_output_to_tensors(output, end_points)
        end_points['center'] = end_points['center_boxnet'] + stage1_center # Bx3

        return end_points, features
    
    def get_depth_feature_op(self, is_training):

        net = tf.expand_dims(self.object_pointcloud, 2)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg1', bn_decay=None)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg2', bn_decay=None)
        net = tf_util.conv2d(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg3', bn_decay=None)
        net = tf_util.conv2d(net, 512, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg4', bn_decay=None)
        net = tf.reduce_max(net, axis = 1)
        
        return net

    def get_depth_feature(self, object_pointcloud):
        
        feed_dict = {self.object_pointcloud:object_pointcloud, self.ops['is_training_pl']:False}
        depth_feature = self.sess.run([self.ops['depth_feature']], feed_dict = feed_dict)
        return depth_feature

    def softmax(self, x):
        ''' Numpy function for softmax'''
        shape = x.shape
        probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
        probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
        return probs

def create_depth_model(model, config_path):
    #Note that folder path must be the folder containing the config.yaml file if omni_camera is True
    if model == 'FPointNet':
        return FPointNet(config_path)
    elif model == 'PointNet':
        return PointNet(config_path)
