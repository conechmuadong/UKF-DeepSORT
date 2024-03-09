# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

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


class UnscentedKalmanFilter:
    def __init__(self):
        ndim = 5
        self._ndim = ndim
        self._no_sigma_points = 2 * ndim + 1
        self._lamda = 3 - (ndim + 2)
        self._update_mat = np.eye(2, ndim)
        self._std_weight_position = 1. / 80
        self._std_weight_velocity = 1. / 600
        self._std_weight_acceleration = 1. / 800
        self.height = 0
        self.sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim + 2))

    def initiate(self, measurement):
        mean_pos = measurement[:2]
        mean_vel = 0
        mean_yaw = np.zeros(2)
        mean = np.r_[mean_pos, mean_vel, mean_yaw]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_velocity * measurement[3],
            1e-6,
            1e-9,
        ]
        self.height = measurement[3]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def generate_sigma_point(self, mean, covariance):
        sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim + 2))
        sigma_points[0] = mean
        L = np.linalg.cholesky(covariance)
        for i in range(0, self._ndim + 2):
            sigma_points[i + 1] = mean + np.sqrt(self._ndim + 2 + self._lamda) * L[i]
            sigma_points[i + 1 + self._ndim + 2] = mean - np.sqrt(self._ndim + 2 + self._lamda) * L[i]
        return sigma_points

    def augmentation(self, mean, covariance):
        mean_aug = np.zeros(7)
        mean_aug[:5] = mean
        covariance_aug = np.zeros((7, 7))
        covariance_aug[:5, :5] = covariance
        std = [
            self._std_weight_acceleration * self.height,
            1e-6
        ]
        covariance_aug[5:, 5:] = np.diag(np.square(std))
        return mean_aug, covariance_aug

    def predict(self, mean, covariance):
        mean, covariance = self.augmentation(mean, covariance)
        sigma_points = self.generate_sigma_point(mean, covariance)
        predicted_sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim))
        for i in range(self._no_sigma_points + 4):
            x = sigma_points[i]
            if x[4] < 1e-20:
                x[0] += x[2] * np.cos(x[3]) + 0.5 * x[5] * np.cos(x[3])
                x[1] += x[2] * np.sin(x[3]) + 0.5 * x[5] * np.sin(x[3])
                x[3] += x[4] + 0.5 * x[6]
            else:
                x[0] += x[2] / x[4] * (np.sin(x[3] + x[4]) - np.sin(x[3])) + 0.5 * x[5] * np.cos(x[3])
                x[1] += x[2] / x[4] * (-np.cos(x[3] + x[4]) + np.cos(x[3])) + 0.5 * x[5] * np.sin(x[3])
                x[3] += x[4] + 0.5 * x[6]
            x[4] += x[6]
            x[2] += x[5]
            predicted_sigma_points[i] = x[:5]

        weights = np.zeros(self._no_sigma_points + 4)
        weights[0] = self._lamda / (self._ndim + 2 + self._lamda)
        weights[1:] = 0.5 / (self._ndim + 2 + self._lamda)
        mean = np.dot(weights, predicted_sigma_points)

        weights = np.diag(weights)
        covariance = np.linalg.multi_dot(
            ((mean.T - predicted_sigma_points).T, weights, (mean.T - predicted_sigma_points)))
        return mean, covariance, predicted_sigma_points

    def project(self, mean, covariance, height, predicted_sigma_points):
        projected_sigma_points = predicted_sigma_points[:, :2].copy()

        weights = np.zeros(self._no_sigma_points + 4)
        weights[0] = self._lamda / (self._ndim+2 + self._lamda)
        weights[1:] = 0.5 / (self._ndim + 2 + self._lamda)
        projected_mean = np.dot(weights, projected_sigma_points)

        weights = np.diag(weights)
        covariance = np.linalg.multi_dot(
            ((projected_mean.T - projected_sigma_points).T, weights, (projected_mean.T - projected_sigma_points)))
        std = [
            self._std_weight_position * height,
            self._std_weight_position * height,
        ]
        innovation_cov = np.diag(np.square(std))
        correlation = np.linalg.multi_dot(
            ((mean.T - predicted_sigma_points).T, weights, (projected_mean.T - projected_sigma_points)))
        return projected_mean, covariance + innovation_cov, correlation

    def update(self, mean, covariance, measurement, predicted_sigma_points):
        projected_mean, projected_covariance, correlation = self.project(mean, covariance,
                                                                         measurement[3], predicted_sigma_points)
        kalman_gain = np.linalg.multi_dot((correlation, np.linalg.inv(projected_covariance)))

        innovation = measurement[:2] - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_covariance, kalman_gain.T))

        if (np.abs(new_mean[0] - measurement[0]) > 1 / 2 * np.abs(innovation[0])
                or np.abs(new_mean[1] - measurement[1]) > 2 / 3 * np.abs(innovation[1])):
            new_mean[0] = 1 / 3 * new_mean[0] + 2 / 3 * measurement[0]
            new_mean[1] = 1 / 3 * new_mean[1] + 2 / 3 * measurement[1]
        self.height = measurement[3]
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, height, predicted_sigma_points,
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
        mean, covariance, _ = self.project(mean, covariance, height, predicted_sigma_points)
        measurements = measurements[:, :2]
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
