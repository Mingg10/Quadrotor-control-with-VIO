import numpy as np
from scipy.spatial.transform import Rotation


class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2
        self.k = self.k_drag/self.k_thrust

        # PID tuning parameters
        self.Kp = np.diag([2.45, 2.45, 10.5])  #position proportional gain
        self.Kd = np.diag([5.25, 5.25, 6.3])  #position derivative gain
        self.Kr = np.diag([1750, 1750, 140])  #angular derivative gain
        self.Kw = np.diag([175, 175, 105])   #angular proportional gain

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """

        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # u1 Calculation
        R = Rotation.from_quat(state['q']).as_matrix()  # calculation for rotational matrix
        error_x = state['x'] - flat_output['x']
        error_v = state['v'] - flat_output['x_dot']
        a_des = flat_output[
                    'x_ddot'] - self.Kd @ error_v - self.Kp @ error_x  # calculation for desired acceleration (3,)
        F_des = self.mass * (a_des + np.array([0, 0, self.g]))  # calculation for desired force (3,)
        cmd_thrust = max(0, np.dot(F_des, R[:, 2]))  # calculation for desired thrust

        # u2 Calculation
        b3_des = F_des / np.linalg.norm(F_des)  # (3,)
        yaw_vec = np.array([np.cos(flat_output['yaw']), np.sin(flat_output['yaw']), 0])
        b2_des = np.cross(b3_des, yaw_vec)
        b2_des /= np.linalg.norm(b2_des)
        b1_des = np.cross(b2_des, b3_des)
        # print(b1_des, b2_des, b3_des)
        R_des = np.column_stack((b1_des, b2_des, b3_des))  # calculation for desired rotational matrix (3,3)
        # print(R_des)
        error_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array(
            [error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])  # calculation for orientation error (3,)

        e_w = state['w']  # calculation for angular velocity error with desired angular velocity set to zero
        cmd_moment = self.inertia @ (-self.Kr @ e_R - self.Kw @ e_w)  # calculation for desired moment

        # quaternion calculation
        cmd_q = Rotation.from_matrix(R_des).as_quat()

        # motor speed calculation
        tau_x, tau_y, tau_z = cmd_moment
        A_inv = np.linalg.inv(np.array([
            [1, 1, 1, 1],
            [0, self.arm_length, 0, -self.arm_length],
            [-self.arm_length, 0, self.arm_length, 0],
            [self.k, -self.k, self.k, -self.k]
        ]))
        motor_thrusts = A_inv @ np.concatenate(([cmd_thrust], cmd_moment))
        motor_thrusts = np.clip(motor_thrusts, 0, None)
        cmd_motor_speeds = np.sqrt(motor_thrusts / self.k_thrust)

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'cmd_q': cmd_q}

        return control_input


