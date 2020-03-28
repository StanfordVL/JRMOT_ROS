# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
import pdb

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

chi2inv90 = {
    1: 2.706,
    2: 4.605,
    3: 6.251,
    4: 7.779,
    5: 9.236,
    6: 10.645,
    7: 12.017,
    8: 13.363,
    9: 14.684}

chi2inv975 = {
    1: 5.025,
    2: 7.378,
    3: 9.348,
    4: 11.143,
    5: 12.833,
    6: 14.449,
    7: 16.013,
    8: 17.535,
    9: 19.023}

chi2inv10 = {
    1: .016,
    2: .221,
    3: .584,
    4: 1.064,
    5: 1.610,
    6: 2.204,
    7: 2.833,
    8: 3.490,
    9: 4.168}


chi2inv995 = {
    1: 0.0000393,
    2: 0.0100,
    3: .0717,
    4: .207,
    5: .412,
    6: .676,
    7: .989,
    8: 1.344,
    9: 1.735}


chi2inv75 = {
    1: 1.323,
    2: 2.773,
    3: 4.108,
    4: 5.385,
    5: 6.626,
    6: 7.841,
    7: 9.037,
    8: 10.22,
    9: 11.39}

def squared_mahalanobis_distance(mean, covariance, measurements):
    # cholesky factorization used to solve for 
    # z = d * inv(covariance)
    # so z is also the solution to 
    # covariance * z = d       
    d = measurements - mean
    # cholesky_factor = np.linalg.cholesky(covariance)
    # z = scipy.linalg.solve_triangular(
    #     cholesky_factor, d.T, lower=True, check_finite=False,
    #     overwrite_b=True)

    squared_maha = np.linalg.multi_dot([d, np.linalg.inv(covariance),
                                        d.T]).diagonal()
    return squared_maha


class EKF(object):
    """
    Generic extended kalman filter class

    """

    def __init__(self):
        pass

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the new track. 
            Unobserved velocities are initialized to 0 mean.

        """
        pass


    def predict_mean(self, mean):
        # Updates predicted state from previous state (function g)
        # Calculates motion update Jacobian (Gt)
        # Returns (g(mean), Gt)
        pass

    def get_process_noise(self, mean, covariance):
        # Returns Rt the motion noise covariance
        pass
    def predict_covariance(self, mean, covariance):
        pass

    def project_mean(self, mean):
        # Measurement prediction from state (function h)
        # Calculations sensor update Jacobian (Ht)
        # Returns (h(mean), Ht)
        pass
    def project_cov(self, mean, covariance):
        pass

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # Perform prediction
        covariance = self.predict_covariance(mean, covariance) 
        mean = self.predict_mean(mean)

        return mean, covariance
    def get_innovation_cov(self, covariance):
        pass

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector 
        covariance : ndarray
            The state's covariance matrix

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """

        # Measurement uncertainty scaled by estimated height
        return self.project_mean(mean), self.project_cov(mean, covariance)

    def update(self, mean, covariance, measurement_t, marginalization=None, JPDA=False):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        predicted_measurement, innovation_cov  = self.project(mean, covariance)
        # cholesky factorization used to solve for kalman gain since
        # K = covariance * update_mat.T * inv(innovation_cov)
        # so K is also the solution to 
        # innovation_cov * K = covariance * update_mat.T
        try:
            chol_factor, lower = scipy.linalg.cho_factor(
                innovation_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(covariance, self._observation_mat.T).T,
                check_finite=False).T
        except:
            # in case cholesky factorization fails, revert to standard solver
            kalman_gain = np.linalg.solve(innovation_cov, np.dot(covariance, self._observation_mat.T).T).T

        if JPDA:
            # marginalization
            innovation = np.zeros((self.ndim)) 
            cov_soft = np.zeros((self.ndim, self.ndim))

            for measurement_idx, measurement in enumerate(measurement_t):

                p_ij = marginalization[measurement_idx + 1] # + 1 for dummy
                y_ij = measurement - predicted_measurement
                innovation += y_ij * p_ij
                cov_soft += p_ij * np.outer(y_ij, y_ij)

            cov_soft = cov_soft - np.outer(innovation, innovation)

            P_star = covariance - np.linalg.multi_dot((
                kalman_gain, innovation_cov, kalman_gain.T))

            p_0 = marginalization[0]
            P_0 = p_0 * covariance + (1 - p_0) * P_star

            new_covariance = P_0 + np.linalg.multi_dot((kalman_gain, cov_soft, kalman_gain.T))
            
        else:
            innovation = measurement_t - predicted_measurement

            new_covariance = covariance - np.linalg.multi_dot((
                kalman_gain, innovation_cov, kalman_gain.T))

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        return new_mean, new_covariance
