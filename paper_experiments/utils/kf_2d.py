# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
import utils.EKF as EKF
import pdb
np.set_printoptions(precision=4, suppress=True)

class KalmanFilter2D(EKF.EKF):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, w, h, vx, vy, vw, vh

    contains the bounding box center position (x, y), width w, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, w, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self, pos_weight, velocity_weight, std_process, std_measurement, initial_uncertainty, gate_limit):
        ndim, dt = 4, 1.
        self.ndim = ndim
        self.img_center = 1242
        # Create Kalman filter model matrices.
        # Motion model is constant velocity, i.e. x = x + Vx*dt
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        # Sensor model is direct observation, i.e. x = x
        self._observation_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_process = std_process
        self._std_weight_measurement = std_measurement
        self._std_weight_pos = pos_weight
        self._std_weight_vel = velocity_weight
        self._initial_uncertainty = initial_uncertainty
        self.LIMIT = gate_limit

    def initiate(self, measurement, flow):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        if flow is not None:
            vel = np.mean(np.reshape(flow[int(mean_pos[1]):int(mean_pos[1]+mean_pos[3]), 
                    int(mean_pos[0]):int(mean_pos[0]+mean_pos[2]), :], (-1, 2)), axis=0)
            mean_vel[:2] = vel
        mean = np.r_[mean_pos, mean_vel]

        # Initialize covariance based on w, h and configured std
        std = [
            (1 + abs(mean_pos[0]/self.img_center - 1)) * self._std_weight_pos * measurement[2],
            (1 + abs(mean_pos[0]/self.img_center - 1)) * self._std_weight_pos * measurement[3],
            (1 + abs(mean_pos[0]/self.img_center - 1)) * self._std_weight_pos * measurement[2],
            (1 + abs(mean_pos[0]/self.img_center - 1)) * self._std_weight_pos * measurement[3],

            (1 + 1.5*abs(mean_pos[0]/self.img_center - 1)) * self._std_weight_vel * measurement[2],
            (1 + 1.5*abs(mean_pos[0]/self.img_center - 1)) * self._std_weight_vel * measurement[3],
            (1 + 1.5*abs(mean_pos[0]/self.img_center - 1)) * self._std_weight_vel * measurement[2],
            (1 + 1.5*abs(mean_pos[0]/self.img_center - 1)) * self._std_weight_vel * measurement[3]]

        covariance = np.diag(np.square(std))*(self._initial_uncertainty*self._std_weight_process)**2
        return mean, covariance

    def predict_mean(self, mean):
        # Updates predicted state from previous state (function g)
        # Calculates motion update Jacobian (Gt)
        # Returns (g(mean), Gt)
        return np.dot(self._motion_mat, mean)
    
    def predict_covariance(self, mean, covariance, last_detection, next_to_last_detection):
        # Updates predicted state from previous state (function g)
        # Calculates motion update Jacobian (Gt)
        # Returns (g(mean), Gt)
        process_noise = self.get_process_noise(mean, last_detection, next_to_last_detection)
        return (np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) 
                     + process_noise)

    def get_process_noise(self, mean, last_detection, next_to_last_detection):
        # Returns Rt the motion noise covariance

        depth_scale = 1
        if last_detection.box_3d is not None:
            dist = last_detection.get_3d_distance()
            depth_scale = max(1,1+(16-dist)/10)
            if next_to_last_detection is not None and next_to_last_detection.box_3d is not None:
                b1 = last_detection.box_3d
                b2 = next_to_last_detection.box_3d
                vel = ((b1[0]-b2[0])**2 + (b1[2]-b2[2])**2)**(1/2)
                if vel > 2: # Fast moving (car) nearby, increase uncertainty
                    depth_scale *= 2
                    pass
                # print(vel)
            # print(dist, depth_scale)

        depth_scale = 1
        # depth_scale *= max(1, 1+(40-mean[2])/50, 1+(40-mean[3])/50) # Note: Scales up small boxes bc higher uncertainty

        # Motion uncertainty scaled by estimated height
        std_pos = [
            depth_scale * self._std_weight_pos * mean[2],
            depth_scale * self._std_weight_pos * mean[3],
            depth_scale * self._std_weight_pos * mean[2],
            depth_scale * self._std_weight_pos * mean[3]]
        std_vel = [
            depth_scale * self._std_weight_vel * mean[2],
            depth_scale * self._std_weight_vel * mean[3],
            depth_scale * self._std_weight_vel * mean[2],
            depth_scale * self._std_weight_vel * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))*self._std_weight_process**2

        return motion_cov

    def project_mean(self, mean):
        # Measurement prediction from state (function h)
        # Calculations sensor update Jacobian (Ht)
        # Returns (h(mean), Ht)
        return np.dot(self._observation_mat, mean)



    def get_measurement_noise(self, measurement):
        # Returns Qt the sensor noise covariance
                
        # Measurement uncertainty scaled by estimated height
        std = [
                self._std_weight_pos*measurement[2],
                self._std_weight_pos*measurement[3],
                self._std_weight_pos*measurement[2],
                self._std_weight_pos*measurement[3]]
        innovation_cov = np.diag(np.square(std))*self._std_weight_measurement**2
        return innovation_cov
    
    def project_cov(self, mean, covariance):
        # Returns S the innovation covariance (projected covariance)
                
        measurement_noise = self.get_measurement_noise(mean)
        innovation_cov = (np.linalg.multi_dot((self._observation_mat, covariance,
                                          self._observation_mat.T))
                     + measurement_noise)
        return innovation_cov

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, use_3d=False):
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
        projected_mean, projected_covariance = self.project(mean, covariance)
        if only_position:
            projected_mean, projected_covariance = projected_mean[:2], projected_covariance[:2, :2]
            measurements = measurements[:, :2]
        max_val = np.amax(projected_covariance)
        # LIMIT = max(mean[2], mean[3]) #*(1 + abs(3*mean[0]/self.img_center - 1))
        # print(projected_covariance)
        if max_val > self.LIMIT:
            projected_covariance *= self.LIMIT / max_val
        return EKF.squared_mahalanobis_distance(projected_mean, projected_covariance, measurements)

class RandomWalkKalmanFilter2D(KalmanFilter2D): #TODO UPDATE THIS DOCUMENTATION
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, w, h

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """
    def __init__(self, pos_weight, velocity_weight, std_process, std_measurement, initial_uncertainty, img_center=1242):
        ndim, dt = 4, 1.
        self.ndim = ndim
        self.img_center = img_center
        # Create Kalman filter model matrices.
        # Motion model is constant velocity, i.e. x = x + Vx*dt
        self._motion_mat = np.eye(2*ndim, 2*ndim)
        self._motion_mat[ndim:, ndim:] = 0
        # Sensor model is direct observation, i.e. x = x
        self._observation_mat = np.eye(ndim, 2*ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_process = std_process
        self._std_weight_measurement = std_measurement
        self._std_weight_pos = pos_weight
        self._std_weight_vel = velocity_weight
        self._initial_uncertainty = initial_uncertainty
