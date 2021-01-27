import random
import numpy as np
import scipy.linalg
import EKF
import pdb
import kf_2d
import os
import pickle
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from read_detections import read_ground_truth_3d_detections, read_ground_truth_2d_detections
np.set_printoptions(precision=4, suppress=True)
from calibration import Calibration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.distances import iou_matrix

class KF_3D(kf_2d.KalmanFilter2D):
    """
    3D Kalman Filter that tracks objets in 3D space

        The 8-dimensional state space

            x, y, z, l, h, w, theta, vx, vz

        contains the bounding box center position (x, z), the heading angle theta, the
        box dimensions l, w, h, and the x and z velocities.

        Object motion follows a constant velocity model. The bounding box location
        (x, y) is taken as direct observation of the state space (linear
        observation model).
    """
    def __init__(self, calib, pos_weight_3d, pos_weight, velocity_weight, theta_weight, 
                    std_process, std_measurement_2d, std_measurement_3d, 
                    initial_uncertainty, omni = True, debug=True):
        self.ndim, self.dt = 9, 1.

        # Create Kalman filter model matrices.
        # Motion model is constant velocity, i.e. x = x + Vx*dt
        self._motion_mat = np.eye(self.ndim, self.ndim)
        self._motion_mat[0, 7] = self.dt
        self._motion_mat[2, 8] = self.dt
        # Sensor model is direct observation, i.e. x = x
        self._observation_mat = np.eye(self.ndim - 2, self.ndim)
        if omni:
            self.x_constant = calib.img_shape[2]/(2*np.pi)
            self.y_constant = calib.median_focal_length_y
            self.calib = calib
        else:
            self.projection_matrix = calib.P

        self.omni = omni
        self._std_weight_pos_3d = pos_weight_3d
        self._std_weight_pos = pos_weight
        self._std_weight_vel = velocity_weight
        self._std_weight_theta= theta_weight

        self._std_weight_process = std_process
        self._initial_uncertainty = initial_uncertainty
        self._std_weight_measurement_2d = std_measurement_2d
        self._std_weight_measurement_3d = std_measurement_3d
        self.debug = debug

    def initiate(self, measurement_3d):

        mean_pos = measurement_3d
        mean_vel = np.zeros((2,))
        mean = np.r_[mean_pos, mean_vel]
        std = [
                self._std_weight_pos_3d * measurement_3d[0],
                self._std_weight_pos_3d * measurement_3d[1],
                self._std_weight_pos_3d * measurement_3d[2],
                self._std_weight_pos_3d * measurement_3d[3],
                self._std_weight_pos_3d * measurement_3d[4],
                self._std_weight_pos_3d * measurement_3d[5],
                self._std_weight_theta,
                self._std_weight_vel,
                self._std_weight_vel]
        covariance = np.diag(np.square(std))*(self._initial_uncertainty*self._std_weight_process)**2
        
        return mean, covariance
        
    def get_process_noise(self, mean):
        std_pos = [
            self._std_weight_pos_3d, # x
            self._std_weight_pos_3d, # y
            self._std_weight_pos_3d, # z
            self._std_weight_pos_3d, # l
            self._std_weight_pos_3d, # h            
            self._std_weight_pos_3d, # w
            self._std_weight_theta # theta
            ]
        std_vel = [
            self._std_weight_vel, # x
            self._std_weight_vel, # z
            ]
        self._motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))*self._std_weight_process**2
        return motion_cov
    
    def get_2d_measurement_noise(self, measurement_2d):
        # Returns Qt the sensor noise covariance
                
        # Measurement uncertainty scaled by estimated height
        std = [
                self._std_weight_pos*measurement_2d[2],
                self._std_weight_pos*measurement_2d[3],
                self._std_weight_pos*measurement_2d[2],
                self._std_weight_pos*measurement_2d[3]]
        innovation_cov = np.diag(np.square(std))*self._std_weight_measurement_2d**2
        return innovation_cov
    
    def get_3d_measurement_noise(self, measurement):
        # Returns Qt the sensor noise covariance
                
        # Measurement uncertainty scaled by estimated height
        std = [
            self._std_weight_pos_3d * measurement[0], # x
            self._std_weight_pos_3d * measurement[1], # y
            self._std_weight_pos_3d * measurement[2], # z
            self._std_weight_pos_3d * measurement[3], # l
            self._std_weight_pos_3d * measurement[4], # h
            self._std_weight_pos_3d * measurement[5], # w
            self._std_weight_theta # theta
            ]
        innovation_cov = np.diag(np.square(std))*self._std_weight_measurement_3d**2
        return innovation_cov
    
    def gating_distance(self, mean, covariance, measurements,
                        only_position=False,
                        use_3d=True):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        if not use_3d:
            corner_points, corner_points_3d = self.calculate_corners(mean)
            H_2d = self.get_2d_measurement_matrix(mean, corner_points, corner_points_3d)
            min_x, min_y = np.amin(corner_points, axis = 0)[:2]
            max_x, max_y = np.amax(corner_points, axis = 0)[:2]
            cov = self.project_cov_2d(mean, covariance, H_2d)
            mean = np.array([min_x, min_y, max_x - min_x, max_y - min_y])
        else:
            mean, cov = mean[:7], covariance[:7, :7]
        if only_position:
            if use_3d:
                mean, cov = mean[:3], cov[:3, :3]
                measurements = measurements[:, :3]
            else:
                mean, cov = mean[:2], cov[:2, :2]
                measurements = measurements[:, :2]
        return EKF.squared_mahalanobis_distance(mean, cov, measurements)

    def project_cov(self, mean, covariance):
        # Returns S the innovation covariance (projected covariance)
                
        measurement_noise = self.get_3d_measurement_noise(mean)
        innovation_cov = (np.linalg.multi_dot((self._observation_mat, covariance,
                                          self._observation_mat.T))
                     + measurement_noise)
        return innovation_cov

    def project_cov_2d(self, mean, covariance, H_2d):
        # Returns S the innovation covariance (projected covariance)
                
        measurement_noise = self.get_2d_measurement_noise(mean)
        innovation_cov = (np.linalg.multi_dot((H_2d, covariance,
                                          H_2d.T))
                     + measurement_noise)
        return innovation_cov
    # @profile
    def update(self, mean, covariance, measurement_2d, measurement_3d = None, marginalization=None, JPDA=False):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (9 dimensional).
        covariance : ndarray
            The state's covariance matrix (9x9 dimensional).
        measurement_2d : ndarray
            The 4 dimensional measurement vector (x, y, w, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        measurement_3d : ndarray
            The 7 dimensional measurement vector (x, y, z, l, h, w, theta), where (x, y, z)
            is the center bottom of the box, l, q, h are the dimensions of the bounding box
            theta is the orientation angle w.r.t. the positive x axis.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """

        if np.any(np.isnan(mean)):
            return mean, covariance
        out_cov = covariance
        H_3d = self._observation_mat
        do_3d = True
        covariance_3d = None
        for meas in measurement_3d:
            if meas is None:
                do_3d = False
                break
        if do_3d:
            S_matrix = self.project_cov(mean, out_cov)
            try:
                chol_factor, lower = scipy.linalg.cho_factor(
                    S_matrix, lower=True, check_finite=False)
                kalman_gain = scipy.linalg.cho_solve(
                    (chol_factor, lower), np.dot(out_cov, H_3d.T).T,
                    check_finite=False).T
            except:
                # in case cholesky factorization fails, revert to standard solver
                kalman_gain = np.linalg.multi_dot((out_cov, H_3d.T, np.linalg.inv(S_matrix)))
            out_cov -= np.linalg.multi_dot((kalman_gain, S_matrix, kalman_gain.T))
            if JPDA:
                innovation_3d = 0
                cov_uncertainty_3d = 0
                for i, detection_3d in enumerate(measurement_3d):
                    innovation_partial = detection_3d - mean[:7]
                    innovation_3d += innovation_partial * marginalization[i+1]
                    cov_uncertainty_3d += marginalization[i+1] * np.outer(innovation_partial, innovation_partial)
                partial_cov = cov_uncertainty_3d-np.outer(innovation_3d, innovation_3d)
                out_cov *= 1 - marginalization[0]
                out_cov += np.linalg.multi_dot((kalman_gain, partial_cov, kalman_gain.T))
                out_cov += marginalization[0]*covariance
            else:
                out_cov = out_cov - np.linalg.multi_dot((kalman_gain, H_3d, out_cov))
                innovation_3d = measurement_3d - mean[:7]
            mean = mean + np.dot(kalman_gain, innovation_3d)
            post_3d_mean = mean
            covariance_3d = out_cov

        if measurement_2d is not None:
            corner_points, corner_points_3d = self.calculate_corners(mean)
            H_2d = self.get_2d_measurement_matrix(mean, corner_points, corner_points_3d)
            #update based on 2D
            min_x, min_y = np.amin(corner_points, axis = 0)[:2]
            max_x, max_y = np.amax(corner_points, axis = 0)[:2]
            S_matrix = self.project_cov_2d(np.array([min_x, min_y, max_x - min_x, max_y - min_y]), out_cov, H_2d)
            try:
                chol_factor, lower = scipy.linalg.cho_factor(
                    S_matrix, lower=True, check_finite=False)
                kalman_gain = scipy.linalg.cho_solve(
                    (chol_factor, lower), np.dot(out_cov, H_2d.T).T,
                    check_finite=False).T
            except:
                # in case cholesky factorization fails, revert to standard solver
                kalman_gain = np.linalg.multi_dot((out_cov, H_2d.T, np.linalg.inv(S_matrix)))
            out_cov = np.dot(np.eye(*out_cov.shape)-np.dot(kalman_gain, H_2d), out_cov)
            if JPDA:
                innovation_2d = 0
                cov_uncertainty_2d = 0
                for i, detection_2d in enumerate(measurement_2d):
                    innovation_partial = detection_2d[:4] - np.array([min_x, min_y, max_x - min_x, max_y - min_y])
                    innovation_2d += innovation_partial * marginalization[i+1] # +1 to account for dummy node
                    cov_uncertainty_2d += marginalization[i+1] * np.outer(innovation_partial, innovation_partial)
                partial_cov = cov_uncertainty_2d-np.outer(innovation_2d, innovation_2d)
                out_cov *= 1 - marginalization[0]
                out_cov += np.linalg.multi_dot((kalman_gain, partial_cov, kalman_gain.T))
                if covariance_3d is None:
                    out_cov += marginalization[0]*covariance
                else:
                    out_cov += marginalization[0]*covariance_3d                    
            else:
                innovation_2d = measurement_2d[:4] - np.array([min_x, min_y, max_x - min_x, max_y - min_y])
            mean = mean + np.dot(kalman_gain, innovation_2d)
        
        if self.debug:
            return mean, out_cov, post_3d_mean
        return mean, out_cov

    # @profile
    def get_2d_measurement_matrix(self, mean, corner_points, corner_points_3d):

        min_x = np.inf
        min_x_idx = None
        max_x = -np.inf
        max_x_idx = None
        min_y = np.inf
        min_y_idx = None
        max_y = -np.inf
        max_y_idx = None
        for idx, pt in enumerate(corner_points):
            if pt[0] < min_x:
                min_x_idx = idx
                min_x = pt[0]
            if pt[0] > max_x:
                max_x_idx = idx
                max_x = pt[0]
            if pt[1] < min_y:
                min_y_idx = idx
                min_y = pt[1]
            if pt[1] > max_y:
                max_y_idx = idx
                max_y = pt[1]
        if self.omni:
            jac_x = np.dot(self.jacobian_omni(corner_points_3d[min_x_idx])[0], self.corner_jacobian(mean, min_x_idx))
            jac_y = np.dot(self.jacobian_omni(corner_points_3d[min_y_idx])[1], self.corner_jacobian(mean, min_y_idx))
            jac_w = np.dot(self.jacobian_omni(corner_points_3d[max_x_idx])[0], self.corner_jacobian(mean, max_x_idx)) - jac_x
            jac_h = np.dot(self.jacobian_omni(corner_points_3d[max_y_idx])[1], self.corner_jacobian(mean, max_y_idx)) - jac_y
        else:
            jac_x = np.dot(self.jacobian(corner_points_3d[min_x_idx])[0], self.corner_jacobian(mean, min_x_idx))
            jac_y = np.dot(self.jacobian(corner_points_3d[min_y_idx])[1], self.corner_jacobian(mean, min_y_idx))
            jac_w = np.dot(self.jacobian(corner_points_3d[max_x_idx])[0], self.corner_jacobian(mean, max_x_idx)) - jac_x
            jac_h = np.dot(self.jacobian(corner_points_3d[max_y_idx])[1], self.corner_jacobian(mean, max_y_idx)) - jac_y
        jac = np.vstack([jac_x, jac_y, jac_w, jac_h])
        jac = np.hstack([jac, np.zeros((jac.shape[0], 2))])
        return jac 
    # Jacobian for projective transformation
    def jacobian(self, pt_3d):
        den = np.sum(self.projection_matrix[2] * pt_3d)
        dxy = (1 - self.projection_matrix[2] * pt_3d/den) * self.projection_matrix[0:2]/den

        return dxy[:, :3]
    
    def jacobian_omni(self, pt_3d):
        jac = np.zeros((2, 3))
        x, y, z = pt_3d[0], pt_3d[1], pt_3d[2]
        denominator = (x**2 + z**2)
        jac[0, 0] = -self.x_constant*(2*x*(z**2)/denominator)
        jac[0, 0] /= denominator
        jac[0, 2] = self.x_constant*2*z/denominator
        jac[0, 2] *= 1 - (z**2)/denominator

        jac[1, 0] = self.y_constant*x*y/denominator
        jac[1, 1] = -self.y_constant
        jac[1,2] = self.y_constant*z*y/denominator
        jac[1, :] /= np.sqrt(denominator)

        return jac

    def calculate_corners(self, box):
        x,y,z,l,h,w,theta = box[:7]
        pt_3d = []
        x_delta_1 = np.cos(theta)*l/2+np.sin(theta)*w/2
        x_delta_2 = np.cos(theta)*l/2 - np.sin(theta)*w/2
        z_delta_1 = np.sin(theta)*l/2-np.cos(theta)*w/2
        z_delta_2 = np.sin(theta)*l/2+np.cos(theta)*w/2
        pt_3d.append((x+x_delta_1, y + h/2, z+z_delta_1, 1))
        pt_3d.append((x+x_delta_2, y + h/2, z+z_delta_2, 1))
        pt_3d.append((x-x_delta_2, y + h/2, z-z_delta_2, 1))
        pt_3d.append((x-x_delta_1, y + h/2, z-z_delta_1, 1))
        pt_3d.append((x+x_delta_1, y - h/2, z+z_delta_1, 1))
        pt_3d.append((x+x_delta_2, y - h/2, z+z_delta_2, 1))
        pt_3d.append((x-x_delta_2, y - h/2, z-z_delta_2, 1))
        pt_3d.append((x-x_delta_1, y - h/2, z-z_delta_1, 1))
        pts_3d = np.vstack(pt_3d)
        pts_2d = self.project_2d(pts_3d)
        return pts_2d, pts_3d
    
    def corner_jacobian(self, pt_3d, corner_idx):
        _, _, _, l, _, w, theta = pt_3d[:7]
        jac = np.eye(3,7)
        
        jac[1, 4] = 0.5 if corner_idx < 4 else -0.5

        jac[0, 3] = 0.5*np.sin(theta) if corner_idx % 4 < 2 else -0.5*np.sin(theta)
        jac[0, 5] = 0.5*np.cos(theta) if corner_idx % 2 == 0 else -0.5*np.cos(theta)
        
        jac[2, 3] = 0.5*np.cos(theta) if corner_idx%4 < 2 else -0.5*np.cos(theta)
        jac[2, 5] = 0.5*np.sin(theta) if corner_idx%2 == 0 else -0.5*np.sin(theta)

        if corner_idx%4 == 0:
            jac[0, 6] = -np.sin(theta)*l/2 + np.cos(theta)*w/2
            jac[2, 6] = np.cos(theta)*l/2 + np.sin(theta)*w/2
        elif corner_idx%4==1:
            jac[0, 6] = -np.sin(theta)*l/2 - np.cos(theta)*w/2
            jac[2, 6] = np.cos(theta)*l/2 - np.sin(theta)*w/2
        elif corner_idx%4==2:
            jac[0, 6] = +np.sin(theta)*l/2 + np.cos(theta)*w/2
            jac[2, 6] = -np.cos(theta)*l/2 + np.sin(theta)*w/2
        else:
            jac[0, 6] = +np.sin(theta)*l/2 - np.cos(theta)*w/2
            jac[2, 6] = -np.cos(theta)*l/2 - np.sin(theta)*w/2

        return jac

    def project_2d(self, pts_3d):
        if self.omni:
            pts_2d = np.array(self.calib.project_ref_to_image_torch(torch.from_numpy(pts_3d)))
        else:
            pts_2d = np.dot(pts_3d, self.projection_matrix.T)
            pts_2d /= np.expand_dims(pts_2d[:, 2], 1)
        return pts_2d[:, :2]


def swap(detections_3d, iou, idx, swap_prob = 0):
    if random.random() > swap_prob:
        return detections_3d[idx]
    else:
        iou_row = iou[idx]
        iou_row[idx] = -1
        max_idx = np.argmax(iou_row)
        if iou_row[max_idx] > 0.4:
            # print("SWAP")
            return detections_3d[max_idx]
        else:
            return detections_3d[idx]


if __name__ == '__main__':
    seq = '0001'
    gt_path = os.path.join('data','KITTI','sequences', seq, 'gt')
    prob_3d_list = [0.6]
    prob_2d_list = [0.9]
    swap_prob = 0
    std_3d = 0.2
    std_2d = 5
    boxes_3d, ids, frame_3d = read_ground_truth_3d_detections(os.path.join(gt_path, '3d_detections.txt'), None)
    boxes_2d, object_ids, frame_2d = read_ground_truth_2d_detections(os.path.join(gt_path, 'gt.txt'), None, nms_threshold = 1)
    boxes_2d[:,2] -= boxes_2d[:,0]
    boxes_2d[:,3] -= boxes_2d[:,1]
    boxes_3d[:,1] -= boxes_3d[:, 4]/2
    calib = Calibration(os.path.join(os.path.dirname(gt_path), 'calib', seq+'.txt'))
    pos_weight = 0.05
    pos_weight_2d = 0.006
    velocity_weight = 0.0007
    theta_weight = 0.000300
    std_process = 2
    std_measurement_2d = 2.6
    std_measurement_3d = 0.01
    initial_uncertainty = 1
    
    kf = KF_3D(calib, pos_weight, pos_weight_2d, velocity_weight, theta_weight, 
                std_process, std_measurement_2d, std_measurement_3d, 
                initial_uncertainty, omni=False, debug=True)
    final_errors = np.zeros((len(prob_2d_list), len(prob_3d_list)))
    random.seed(14295)
    np.random.seed(14295)
    for idx_3d, prob_3d in enumerate(prob_3d_list):
        for idx_2d, prob_2d in enumerate(prob_2d_list):
            id_means = {idx:[] for idx in np.unique(ids)}
            id_means_2d = {idx:[] for idx in np.unique(ids)}
            id_preds = {idx:[] for idx in np.unique(ids)}
            id_meas = {idx:[] for idx in np.unique(ids)}
            id_errors = {idx:[] for idx in np.unique(ids)}
            for frame in sorted(np.unique(frame_2d)):
                frame_mask = frame_2d==frame
                frame_boxes_2d = boxes_2d[frame_mask]
                frame_boxes_3d = boxes_3d[frame_mask]
                frame_ids = ids[frame_mask]
                iou = 1-iou_matrix(frame_boxes_2d[:,:4], frame_boxes_2d[:,:4], max_iou=10) #output of function is 1 - IoU
                for idx, object_id in enumerate(frame_ids):
                    if frame_boxes_3d[idx][2] > 30:
                        continue
                    noise_2d = np.random.randn(*frame_boxes_2d[idx].shape)*std_2d
                    noise_3d = np.random.randn(*frame_boxes_3d[idx].shape)*std_3d
                    if len(id_means[object_id.item()]) == 0:
                        mean, cov = kf.initiate(frame_boxes_2d[idx]+noise_2d, frame_boxes_3d[idx]+noise_3d)
                        id_means[object_id.item()].append((mean, cov, frame))
                        # id_preds[object_id.item()].append((mean, cov, frame))
                        # id_meas[object_id.item()].append((frame_boxes_3d[idx], frame_boxes_2d[idx], frame))
                        # id_errors[object_id.item()].append((np.sqrt(np.sum((mean[:3] - frame_boxes_3d[idx][:3])**2)), frame))
                        continue
                    mean, cov = kf.predict(id_means[object_id.item()][-1][0], id_means[object_id.item()][-1][1])
                    id_preds[object_id.item()].append((mean, cov, frame))
                    # if object_id.item()==3:
                    #     print("3D box: ", frame_boxes_3d[idx])
                    #     print("Old mean:", id_means[object_id.item()][0])
                    #     print("Predicted mean:", mean)
                        # pdb.set_trace()
                    if random.random() < prob_2d:
                        if random.random() < prob_3d:
                            mean, cov, mean_2d = kf.update(mean, cov, frame_boxes_2d[idx]+noise_2d, swap(frame_boxes_3d, iou, idx, swap_prob)+noise_3d)
                        else:
                            mean, cov, mean_2d = kf.update(mean, cov, frame_boxes_2d[idx]+noise_2d, None)
                    # if object_id.item()==12:
                    #     print("Updated mean after 2D:", mean_2d)
                    #     print("Updated mean after 3D:", mean)
                    #     print("Error:", np.sqrt(np.sum((mean[:3] - frame_boxes_3d[idx][:3])**2)))
                    #     if np.sqrt(np.sum((mean[:3] - frame_boxes_3d[idx][:3])**2)) > 1:
                    #         pdb.set_trace()
                    id_means[object_id.item()].append((mean, cov, frame))
                    id_means_2d[object_id.item()].append((mean_2d, frame))
                    id_meas[object_id.item()].append((frame_boxes_3d[idx], frame_boxes_2d[idx], frame))
                    id_errors[object_id.item()].append((np.sqrt(np.sum((mean[:3] - frame_boxes_3d[idx][:3])**2)), frame))
            errors = [np.mean(error[0]) for idx, error in id_errors.items() if len(error) > 0]
            final_errors[idx_2d, idx_3d] = np.mean(errors)
            print("3D prob: %f %% & 2D prob: %f %% & swap prob: %f %%  RMSE: %f"%(prob_3d*100, prob_2d*100, swap_prob*100, final_errors[idx_2d, idx_3d]))
            # if :
            with open('results/kf_mean_pickle.p', 'wb') as f:
                pickle.dump([id_means, id_means_2d, id_meas, id_preds], f)

    print(final_errors)