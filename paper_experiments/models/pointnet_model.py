import os, pdb
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import configparser
from utils.pointnet_transform_nets import input_transform_net, feature_transform_net
import utils.pointnet_tf_util as pointnet_tf_util


class PointNet():
    def __init__(self, config_path):
        parser = configparser.SafeConfigParser()
        parser.read(config_path)
        num_points = parser.getint('general', 'num_point')
        depth_model_path = parser.get('general', 'depth_model_path')

        with tf.device('/gpu:'+str(0)):
            self.pointclouds_pl, _ = self.placeholder_inputs(1, num_points)
            self.is_training_pl = tf.placeholder(tf.bool, shape=())

            # simple model
            feature = self.get_model(self.pointclouds_pl, self.is_training_pl)
            self.feature = feature
            # Add ops to save and restore all the variables.
        
        self.saver = tf.train.Saver()
        #Create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        #Initialize variables
        self.sess.run(tf.global_variables_initializer())
        #Restore model weights
        self.saver.restore(self.sess, depth_model_path)

    def __call__(self, input_point_cloud):
        feed_dict = {self.pointclouds_pl: input_point_cloud,
                     self.is_training_pl: False}
        features = self.sess.run(self.feature,feed_dict=feed_dict)
        return features

    def placeholder_inputs(self, batch_size, num_point):
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, None, 3))
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
        return pointclouds_pl, labels_pl


    def get_model(self, point_cloud, is_training, bn_decay=None):
        """ Classification PointNet, input is BxNx3, output Bx40 """
        batch_size = point_cloud.get_shape()[0].value
        end_points = {}

        with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        net = pointnet_tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = pointnet_tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)

        with tf.variable_scope('transform_net2', reuse=tf.AUTO_REUSE) as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])

        net = pointnet_tf_util.conv2d(net_transformed, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = pointnet_tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = pointnet_tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf.reduce_max(net, axis = 1)

        net = tf.reshape(net, [batch_size, -1])
        feature = net

        return feature


    def get_loss(self, pred, label, end_points, reg_weight=0.001):
        """ pred: B*NUM_CLASSES,
            label: B, """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
        classify_loss = tf.reduce_mean(loss)
        tf.summary.scalar('classify loss', classify_loss)

        # Enforce the transformation as orthogonal matrix
        transform = end_points['transform'] # BxKxK
        K = transform.get_shape()[1].value
        mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
        mat_diff = mat_diff - tf.constant(np.eye(K), dtype=tf.float32)
        mat_diff_loss = tf.nn.l2_loss(mat_diff) 
        tf.summary.scalar('mat loss', mat_diff_loss)

        return classify_loss + mat_diff_loss * reg_weight
    