# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
import EKF
import pdb


class KalmanFilter3D(EKF.EKF):
    """
    A simple 3D Kalman filter for tracking bounding cuboids in 3d.

    The 12-dimensional state space

        x, y, l, h, w, theta, Vx, Vy, Vl, Vh, Vw, Vtheta

    contains the bounding box center position (x, y), width w, height h,
    length l, heading theta, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    is taken as direct observation of the state space (linear observation model).

    """

    def __init__(self):
        ndim, dt = 6, 1.
        self.ndim = ndim

        # Create Kalman filter model matrices.
        # Motion model is constant velocity, i.e. x = x + Vx*dt
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        # Sensor model is direct observation, i.e. x = x
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_motion_pos = 0.8
        self._std_motion_vel = 0.1
        self._std_motion_theta= 0.017*1 # ~1 degrees
        self._std_motion_omega = 0.017*0.1 # ~0.1 degrees

        self._std_sensor_pos = 0.8
        self._std_sensor_vel = 0.1
        self._std_sensor_theta= 0.017*5 # ~5 degrees

        std_pos = [
            self._std_motion_pos, # x
            self._std_motion_pos, # y
            self._std_motion_pos, # l
            self._std_motion_pos, # h
            self._std_motion_pos, # w            
            self._std_motion_theta # theta
            ]
        std_vel = [
            self._std_motion_vel, # x
            self._std_motion_vel, # y
            self._std_motion_vel, # l
            self._std_motion_vel, # h
            self._std_motion_vel, # w            
            self._std_motion_omega # omega
            ]
        self._motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        std = [
            self._std_sensor_pos, # x
            self._std_sensor_pos, # y
            self._std_sensor_pos, # l
            self._std_sensor_pos, # h
            self._std_sensor_pos, # w
            self._std_sensor_theta # theta
            ]
        self._innovation_cov = np.diag(np.square(std))

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, l, h, w, theta) 

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (12 dimensional) and covariance matrix (12x12
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Initialize covariance 
        std = [ 2,  2,  2,  2,  2,  2,
               10, 10, 10, 10, 10, 10
              ]
        covariance = self._motion_cov * np.diag(np.square(std))
        return mean, covariance

    def motion_update(self, mean, covariance):
        # Updates predicted state from previous state (function g)
        # Calculates motion update Jacobian (Gt)
        # Returns (g(mean), Gt)
        mean = np.dot(self._motion_mat, mean)
        return mean, self._motion_mat

    def get_motion_cov(self, mean, covariance):
        # Returns Rt the motion noise covariance

        return self._motion_cov

    def sensor_update(self, mean, covariance):
        # Measurement prediction from state (function h)
        # Calculations sensor update Jacobian (Ht)
        # Returns (h(mean), Ht)
        mean = np.dot(self._update_mat, mean)
        return mean, self._update_mat

    def get_innovation_cov(self, mean, covariance):
        # Returns Qt the sensor noise covariance
        return self._innovation_cov

    def adjust_angle(self, measured, target):
        step = 2*np.pi
        measured += step*np.round((target - measured)/step)
        return measured

    def update(self, mean, covariance, meas_in, marginalization=None, JPDA=False):
        measurement = np.copy(meas_in)
        if measurement.ndim == 1:
            measurement[5] = self.adjust_angle(measurement[5], mean[5])
        else:
            measurement[:,5] = self.adjust_angle(measurement[:,5], mean[5])
        return EKF.EKF.update(self, mean, covariance, measurement, marginalization, JPDA)

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 6 degrees of
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
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        return EKF.squared_mahalanobis_distance(mean, covariance, measurements)
