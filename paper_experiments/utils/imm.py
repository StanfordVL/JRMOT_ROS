# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
import utils.EKF as EKF
import pdb
import utils.kf_2d as kf_2d
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

class IMMFilter2D(EKF.EKF):
    """
    An IMM filter for tracking bounding boxes in image space.
    Contains 2 Kalman Filters
    Filter 1: Constant Velocity Model:
        The 8-dimensional state space
            x, y, a, h, vx, vy, va, vh
        contains the bounding box center position (x, y), aspect ratio a, height h,
        and their respective velocities.
        Object motion follows a constant velocity model. The bounding box location
        (x, y, a, h) is taken as direct observation of the state space (linear
        observation model).
    Filter 2: Random Walk Model:
        The 4-dimensional state space
            x, y, a, h
        contains the bounding box center position (x, y), aspect ratio a, height h.
        Object motion follows a random walk model. The bounding box location
        (x, y, a, h) is taken as direct observation of the state space (linear
        observation model).
    """
    def __init__(self, kf_vel_params=(1./20, 1./160, 1, 1, 2), kf_walk_params=(1./20, 1./160, 1, 1, 2), markov=(0.9,0.7)):
        self.kf1 = kf_2d.KalmanFilter2D(*kf_vel_params)
        self.kf2 = kf_2d.RandomWalkKalmanFilter2D(*kf_walk_params)

        self.markov_transition = np.asarray([[markov[0], 1-markov[0]],
                                             [markov[1], 1-markov[1]]])

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
            Returns the mean vector (2,8 dimensional) and covariance matrix (2,8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos1, cov1 = self.kf1.initiate(measurement, flow)
        #Random walk does not need the flow
        mean_pos2, cov2 = self.kf2.initiate(measurement, None)

        covariance = np.dstack([cov1, cov2])
        covariance = np.transpose(covariance, axes=(2,0,1))
        mean = np.vstack([mean_pos1, mean_pos2])
        model_probs = np.ones((2,1))*0.5

        return mean, covariance, model_probs


    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
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
        dist1 = self.kf1.gating_distance(mean[0, :], covariance[0, :, :], measurements, only_position)
        dist2 = self.kf2.gating_distance(mean[1, :], covariance[1, :, :], measurements, only_position)
        return np.where(dist1 < dist2, dist1, dist2)

    def update(self, mean, covariance, measurement, model_probabilities, marginalization=None, JPDA=False):
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
        # cholesky factorization used to solve for kalman gain since
        # K = covariance * update_mat * inv(projected_cov)
        # so K is also the solution to 
        # projected_cov * K = covariance * update_mat
        # model_probabilities = np.dot(self.markov_transition.T, model_probabilities)
        # combined_H = np.stack([self.kf1._update_mat, self.kf2._update_mat])
        # S = np.linalg.multi_dot([combined_H, covariance, np.transpose(combined_H, (0,2,1))])

        mean_1, cov_1 = self.kf1.project(mean[0], covariance[0])
        mean_2, cov_2 = self.kf2.project(mean[1], covariance[1])

        distance_1 = EKF.squared_mahalanobis_distance(mean_1, cov_1, measurement)
        distance_2 = EKF.squared_mahalanobis_distance(mean_2, cov_2, measurement)

        distance = np.vstack([distance_1, distance_2])

        distance -= np.amin(distance)
        
        dets = np.vstack([np.sqrt(np.linalg.det(cov_1)), np.sqrt(np.linalg.det(cov_2))])
        if distance.ndim > 1:
            likelihood = np.sum(np.exp(-distance/2)/dets, axis = -1, keepdims = True)
        else:
            likelihood = np.exp(-distance/2)/dets

        model_probs = (likelihood*model_probabilities)/\
                        np.sum(likelihood*model_probabilities)
            

        out_mean_1, out_cov_1 = self.kf1.update(mean[0], covariance[0], measurement, marginalization, JPDA)
        out_mean_2, out_cov_2 = self.kf2.update(mean[1], covariance[1], measurement, marginalization, JPDA)
        out_mean = np.vstack([out_mean_1, out_mean_2])
        out_cov = np.dstack([out_cov_1, out_cov_2])
        out_cov = np.transpose(out_cov, axes=(2,0,1))

        return out_mean, out_cov, model_probs
    
    def predict(self, mean, covariance, model_probabilities):
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

        model_future_probabilities = np.dot(self.markov_transition.T, model_probabilities)
        model_transition_probabilities = self.markov_transition*(model_probabilities/model_future_probabilities.T)
        mixed_mean_1, mixed_cov_1, mixed_mean_2, mixed_cov_2 = self.mix_models(mean[0], covariance[0], mean[1], covariance[1], model_transition_probabilities)
        out_mean_1, out_cov_1 = self.kf1.predict(mixed_mean_1, mixed_cov_1)
        out_mean_2, out_cov_2 = self.kf2.predict(mixed_mean_2, mixed_cov_2)

        out_mean = np.vstack([out_mean_1, out_mean_2])
        out_cov = np.dstack([out_cov_1, out_cov_2])
        out_cov = np.transpose(out_cov, axes=(2,0,1))
        return out_mean, out_cov, model_future_probabilities

    def mix_models(self, mean_1, cov_1, mean_2, cov_2, model_transition_probabilities):
        
        mixed_mean_1 = model_transition_probabilities[0, 0]*mean_1 + model_transition_probabilities[1, 0]*mean_2
        mixed_mean_2 = model_transition_probabilities[0, 1]*mean_1 + model_transition_probabilities[1, 1]*mean_2
        mean_diff_12 = mean_1 - mixed_mean_2
        mean_diff_21 = mean_2 - mixed_mean_1
        mean_diff_11 = mean_1 - mixed_mean_1
        mean_diff_22 = mean_2 - mixed_mean_2

        mixed_cov_1 = model_transition_probabilities[0, 0]*(cov_1+np.outer(mean_diff_11, mean_diff_11)) + \
                        model_transition_probabilities[1, 0]*(cov_2+np.outer(mean_diff_21, mean_diff_21))
        mixed_cov_2 = model_transition_probabilities[0, 1]*(cov_2+np.outer(mean_diff_12, mean_diff_12)) + \
                    model_transition_probabilities[1, 1]*(cov_2+np.outer(mean_diff_22, mean_diff_22))

        return mixed_mean_1, mixed_cov_1, mixed_mean_2, mixed_cov_2
    
    @staticmethod
    def combine_states(mean, cov, model_probabilities):

        mean = np.sum(model_probabilities*mean, axis = 0)
        covariance = np.sum(np.expand_dims(model_probabilities,2)*cov, axis = 0)

        return mean, covariance
        
def generate_particle_motion(motion_matrices, initial_state, process_noise, length = 100):
    state_list = [initial_state]
    seed_mode = 0 if np.random.random() < 0.5 else 1
    markov_transition_matrix = np.asarray([[0.9, 0.1],[.7, 0.3]])
    modes = [seed_mode]
    for i in range(length):
        modes.append(seed_mode)
        motion_matrix = motion_matrices[seed_mode]
        state_list.append(np.dot(motion_matrix, state_list[-1])+np.random.randn(*initial_state.shape)*process_noise[seed_mode])
        if np.random.rand() < markov_transition_matrix[seed_mode][0]:
            seed_mode = 0
        else:
            seed_mode = 1
    return np.array(state_list), modes

def generate_observations(input_state_list, observation_matrix, observation_noise):
    observation_shape = np.dot(observation_matrix, input_state_list[0]).shape
    output = [np.dot(observation_matrix, state)+np.random.randn(*observation_shape)*observation_noise 
                for state in input_state_list]
    return np.array(output)

if __name__=='__main__':
    imm_filter = IMMFilter2D()
    motion_matrix = np.eye(8)
    motion_matrix[0,4] = 1
    motion_matrix[1,5] = 1
    initial_state = np.array([0,0,1,1,1,1,0,0])
    states, modes = generate_particle_motion([motion_matrix, np.eye(8)], initial_state, [0.1, 2], 50)
    plt.subplot(211)
    plt.plot(states[:,0], states[:,1], linestyle = '--', marker='.', label= 'True state')
    observation_matrix = np.eye(4,8)
    obs = generate_observations(states, observation_matrix, 0.5)
    # plt.scatter(obs[:,0], obs[:,1], marker='x', color='green', label = 'observation')
    rnd_filter = kf_2d.KalmanFilter2D()
    mean, covariance, probs = imm_filter.initiate(obs[0])
    mean_rand, cov_rand = rnd_filter.initiate(obs[0])
    mean_list, covariance_list, probs_list = [], [], []
    mean_list_rand, covariance_list_rand = [], []

    combined_mean, combined_cov = imm_filter.combine_states(mean, covariance, probs)
    mean_list.append(combined_mean)
    covariance_list.append(combined_cov)
    mean_list_rand.append(mean_rand)
    covariance_list_rand.append(cov_rand)
    probs_list.append(probs)
    for idx, i in enumerate(obs[1:]):
        mean_rand_new, cov_rand_new = rnd_filter.predict(mean_rand, cov_rand)
        mean_rand, cov_rand = rnd_filter.update(mean_rand_new, cov_rand_new, i)
        mean_list_rand.append(mean_rand)
        covariance_list_rand.append(cov_rand)


        mean_new, covariance_new, probs_new = imm_filter.predict(mean, covariance, probs)
        mean, covariance, probs = imm_filter.update(mean_new, covariance_new, i, probs_new)
        combined_mean, combined_cov = imm_filter.combine_states(mean, covariance, probs)
        pdb.set_trace()
        pdb.set_trace()
        mean_list.append(combined_mean)
        covariance_list.append(combined_cov)
        probs_list.append(probs)

    mean_list = np.array(mean_list)
    mean_list_rand = np.array(mean_list_rand)
    plt.plot(mean_list[:, 0], mean_list[:, 1], marker='+', c='k', label = 'IMMestimate', alpha = 0.6)
    plt.plot(mean_list_rand[:, 0], mean_list_rand[:, 1], marker=',', c='orange', label = 'CV estimate', alpha = 0.6)
    # plt.scatter(mean_list[:, 0], mean_list[:, 1], marker='+', c=np.vstack([probs, np.zeros((1,1))]).T, label = 'IMMestimate')
    # plt.scatter(mean_list_rand[:, 0], mean_list_rand[:, 1], marker='x', c='orange', label = 'random walk estimate')
    MSE_IMM = np.mean((mean_list[:,:2]-states[:,:2])**2)
    MSE = np.mean((mean_list_rand[:,:2]-states[:,:2])**2)
    print("MSE: %f for 2D filter"%MSE)
    print("MSE: %f for IMM filter"%MSE_IMM)
    plt.legend()
    plt.subplot(212)

    plt.plot(modes, label='True modes')
    plt.plot([i[1] for i in probs_list], label='predicted modes')
    plt.legend()
    plt.show()
