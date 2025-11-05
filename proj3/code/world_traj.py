import numpy as np
from scipy.interpolate import CubicSpline
from graph_search import graph_search

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.speed = 2.5        # default speed
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        # setup globle variables
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.epsilon = 0.2

        # RDP waypoints reduction
        rdp_points = self.rdp(self.path, self.epsilon)
        self.rdp_points = np.array(rdp_points)
        points = [rdp_points[0]]

        for i in range(1, len(self.rdp_points)):
            p0 = np.array(rdp_points[i - 1])
            p1 = np.array(rdp_points[i])
            midpoint = (p0 + p1) / 2
            points.append(midpoint)
        points.append(rdp_points[-1])
        self.points = np.array(points)

        self.t_seq = [0.0]
        for i in range(1, len(self.points)):
            d = np.linalg.norm(self.points[i] - self.points[i - 1])
            if d >= 3.0:
                self.speed = 4.55
            elif d >= 2.0:
                self.speed = 3.5
            else:
                self.speed = 2.5

            dt = d / self.speed
            self.t_seq.append(self.t_seq[-1] + dt)

        self.t_seq = np.array(self.t_seq)
        self.T_total = self.t_seq[-1]

        # cubic spline
        self.spline_x = CubicSpline(self.t_seq, self.points[:, 0])
        self.spline_y = CubicSpline(self.t_seq, self.points[:, 1])
        self.spline_z = CubicSpline(self.t_seq, self.points[:, 2])

    def rdp(self, path_points, threshold):
        if len(path_points) < 3:
            return path_points

        start_point = np.array(path_points[0])
        end_point = np.array(path_points[-1])
        base = end_point - start_point
        norm_base = np.linalg.norm(base)
        max_delta = -1
        ind = -1

        for i in range(1, len(path_points) - 1):
            point = np.array(path_points[i])
            v = start_point - point
            delta = np.linalg.norm(np.cross(base, v)) / norm_base
            if delta > max_delta:
                max_delta = delta
                ind = i

        if max_delta > threshold:
            left = self.rdp(path_points[:ind + 1], threshold)
            right = self.rdp(path_points[ind:], threshold)
            return list(left[:-1]) + list(right)
        else:
            return [path_points[0], path_points[-1]]

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        # time over total trajectory

        if t >= self.T_total:
            x = self.points[-1]
        else:
            t_ = np.clip(t, 0.0, self.T_total)
            x = np.array([self.spline_x(t_), self.spline_y(t_), self.spline_z(t_)])
            x_dot = np.array([self.spline_x.derivative(1)(t_),
                              self.spline_y.derivative(1)(t_),
                              self.spline_z.derivative(1)(t_)])
            x_ddot = np.array([self.spline_x.derivative(2)(t_),
                               self.spline_y.derivative(2)(t_),
                               self.spline_z.derivative(2)(t_)])
            x_dddot = np.array([self.spline_x.derivative(3)(t_),
                                self.spline_y.derivative(3)(t_),
                                self.spline_z.derivative(3)(t_)])

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot,
                       'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output






