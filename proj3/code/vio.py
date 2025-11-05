#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time inteArval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    rot = q.as_matrix()
    delta_theta = (w_m - w_b) * dt
    delta_q = Rotation.from_rotvec(delta_theta.flatten())
    new_p = p + v * dt + 0.5 * (rot @ (a_m - a_b) + g) * dt**2
    new_v = v + (rot @ (a_m - a_b) + g) * dt
    new_q = q * delta_q

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state
    rot = q.as_matrix()
    i_3 = np.eye(3)
    delta_theta = (w_m - w_b) * dt
    def skew(vec):
        x, y, z = vec.flatten()
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])

    # YOUR CODE HERE
    # Construct noise term Fi (18x12)
    f_i = np.zeros((18, 12))
    f_i[3:6, 0:3] = i_3
    f_i[6:9, 3:6] = i_3
    f_i[9:12, 6:9] = i_3
    f_i[12:15, 9:12] = i_3

    # Construct noise term Qi (12x12)
    q_i = np.zeros((12, 12))
    v_i = (accelerometer_noise_density ** 2) * (dt ** 2) * i_3
    theta_i = (gyroscope_noise_density ** 2) * (dt ** 2) * i_3
    a_i = (accelerometer_random_walk ** 2) * dt * i_3
    omega_i = (gyroscope_random_walk ** 2) * dt * i_3
    q_i[0:3, 0:3] = v_i
    q_i[3:6, 3:6] = theta_i
    q_i[6:9, 6:9] = a_i
    q_i[9:12, 9:12] = omega_i

    # Construct system jacobian Fx (18x18)
    f_x = np.eye(18)
    f_x[0:3, 3:6] = i_3 * dt
    f_x[3:6, 6:9] = - rot @ skew(a_m - a_b) * dt
    f_x[3:6, 9:12] = -rot * dt
    f_x[3:6, 15:18] = i_3 * dt
    f_x[6:9, 6:9] = Rotation.from_rotvec(delta_theta.flatten()).as_matrix().T
    f_x[6:9, 12:15] = -i_3 * dt

    # return an 18x18 covariance matrix
    p_old = error_state_covariance
    p_new = f_x @ p_old @ f_x.T + f_i @ q_i @ f_i.T
    return p_new


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state
    rot = q.as_matrix()
    def skew(v):
        x, y, z = v.flatten()
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])

    # YOUR CODE HERE
    # Compute innovation (2x1)
    p_c = rot.T @ (Pw - p)
    x_c, y_c, z_c = p_c.flatten()
    predicted_uv = np.array([[x_c / z_c], [y_c / z_c]])
    innovation = uv - predicted_uv
    inno_mag = np.linalg.norm(innovation)

    # Check if consider outlier
    if inno_mag > error_threshold:
        return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

    # Compute measurement Jacobian Ht (2x18)
    dzt_dpc = np.array([
        [1 / z_c, 0, -x_c / (z_c ** 2)],
        [0, 1 / z_c, -y_c / (z_c ** 2)]
    ])
    dpc_dtheta = skew(p_c)
    dpc_dp = -rot.T
    dzt_dtheta = dzt_dpc @ dpc_dtheta
    dzt_dp = dzt_dpc @ dpc_dp
    h_t = np.zeros((2, 18))
    h_t[:, 0:3] = dzt_dp
    h_t[:, 6:9] = dzt_dtheta

    # Compute Kalman gain Kt (18x2)
    k_t = error_state_covariance @ h_t.T @ np.linalg.inv(h_t @ error_state_covariance @ h_t.T + Q)

    # Update error state
    delta_x = k_t @ innovation
    delta_p = delta_x[0: 3]
    delta_v = delta_x[3: 6]
    delta_theta = delta_x[6: 9]
    delta_a_b = delta_x[9:12]
    delta_w_b = delta_x[12:15]
    delta_g = delta_x[15:18]
    p = p + delta_p
    v = v + delta_v
    q = q * Rotation.from_rotvec(delta_theta.flatten())  # rotation correction
    a_b = a_b + delta_a_b
    w_b = w_b + delta_w_b
    g = g + delta_g

    # Compute new error state covariance (18x18)
    i_18 = np.eye(18)
    sigma_t = (i_18 - k_t @ h_t) @ error_state_covariance @ (i_18 - k_t @ h_t).T + k_t @ Q @ k_t.T
    error_state_covariance = sigma_t

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
