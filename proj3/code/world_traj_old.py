
import numpy as np
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
        # must choose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.22, 0.22, 0.22])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.space_factor = 10
        self.speed = 3.275
        self.spacing = self.space_factor * np.min(self.resolution)
        list_path = self.path.tolist()

        filtered_points = [list_path[0]]  # Start with first point
        last_kept_point = list_path[0]  # Initialize last kept point

        # Process middle points
        for i in range(1, len(list_path) - 1):  # Exclude last point
            if np.linalg.norm(np.array(list_path[i]) - np.array(last_kept_point)) >= self.spacing:
                filtered_points.append(list_path[i])  # Keep this point
                last_kept_point = list_path[i]  # Update last kept point

        filtered_points.append(list_path[-1])       # Add last point
        self.points = np.array(filtered_points)     # Convert back to NumPy array
        n = len(self.points)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.
        # Compute time intervals based on constant speed
        segment_distances = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        segment_durations = segment_distances / self.speed
        self.segment_times = np.zeros(n)
        self.segment_times[1:] = np.cumsum(segment_durations)

        # Solve minimum jerk trajectory coefficients
        self.coefficients = self.compute_minimum_jerk()

    def compute_minimum_jerk(self):
        """
        Solves for minimum jerk polynomial coefficients for each segment.
        """
        num_segments = len(self.points) - 1
        num_constraints = 6 * num_segments  # Number of constraints (matches unknowns)
        coefficients = np.zeros((6 * num_segments, 3))  # (coefficients, xyz)

        # Construct A matrix and B vector
        A = np.zeros((num_constraints, num_constraints))
        B = np.zeros((num_constraints, 3))  # Separate B for x, y, z
        row = 0

        # Boundary Velocity & Acceleration: 4
        t_0 = 0
        t_T = self.segment_times[-1] - self.segment_times[-2]       # Time allocation

        A[row, 0: 6] = [5 * t_0 ** 4, 4 * t_0 ** 3, 3 * t_0 ** 2, 2 * t_0, 1, 0]        # Initial velocity
        B[row] = np.zeros(3)
        row += 1

        A[row, 0: 6] = [20 * t_0 ** 3, 12 * t_0 ** 2, 6 * t_0, 2, 0, 0]        # Initial acceleration
        B[row] = np.zeros(3)
        row += 1

        A[row, -6:] = [5 * t_T ** 4, 4 * t_T ** 3, 3 * t_T ** 2, 2 * t_T, 1, 0]     # Final velocity
        B[row] = np.zeros(3)
        row += 1

        A[row, -6:] = [20 * t_T ** 3, 12 * t_T ** 2, 6 * t_T, 2, 0, 0]        # Final acceleration
        B[row] = np.zeros(3)
        row += 1

        for i in range(num_segments):
            t_0 = 0
            t_T = self.segment_times[i + 1] - self.segment_times[i]
            x_0 = self.points[i]
            x_T = self.points[i + 1]

            # Boundary Positions (Start & End): 2(m-1)
            A[row, i * 6: (i + 1) * 6] = [t_0 ** 5, t_0 ** 4, t_0 ** 3, t_0 ** 2, t_0 ** 1, 1]  # x(0) = x_0
            B[row] = x_0
            row += 1

            A[row, i * 6: (i + 1) * 6] = [t_T ** 5, t_T ** 4, t_T ** 3, t_T ** 2, t_T ** 1, 1]  # x(T) = x_T
            B[row] = x_T
            row += 1

            # Continuity Conditions: 4(m-2)
            if i <= num_segments - 2:
                A[row, i * 6: (i + 1) * 6] = [5 * t_T ** 4, 4 * t_T ** 3, 3 * t_T ** 2, 2 * t_T, 1, 0]  # Velocity continuity
                A[row, (i + 1) * 6: (i + 2) * 6] = [-5 * t_0 ** 4, -4 * t_0 ** 3, -3 * t_0 ** 2, -2 * t_0, -1, 0]
                row += 1

                A[row, i * 6: (i + 1) * 6] = [20 * t_T ** 3, 12 * t_T ** 2, 6 * t_T, 2, 0, 0]  # Acceleration continuity
                A[row, (i + 1) * 6: (i + 2) * 6] = [-20 * t_0 ** 3, -12 * t_0 ** 2, -6 * t_0, -2, 0, 0]
                row += 1

                A[row, i * 6: (i + 1) * 6] = [60 * t_T ** 2, 24 * t_T, 6, 0, 0, 0 ]  # Jerk continuity
                A[row, (i + 1) * 6: (i + 2) * 6] = [-60 * t_0 ** 2, -24 * t_0, -6, 0, 0, 0]
                row += 1

                A[row, i * 6: (i + 1) * 6] = [120 * t_T, 24, 0, 0, 0, 0]  # Snap continuity
                A[row, (i + 1) * 6: (i + 2) * 6] = [-120 * t_0, -24, 0, 0, 0, 0]
                row += 1

        # Solve Ax = B
        A += np.eye(A.shape[0]) * 1e-6  # Small perturbation to avoid singular matrix
        coefficients = np.linalg.inv(A) @ B     # (6m-6, 3) numpy array of coefficients

        return coefficients

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
        # Compute normalized time within segment
        segment_idx = np.searchsorted(self.segment_times, t) - 1
        segment_idx = np.clip(segment_idx, 0, len(self.segment_times) - 1)

        t_0 = self.segment_times[segment_idx]
        t_end = self.segment_times[-1]
        if t >= self.segment_times[-1]:
            x = self.points[-1]  # Hold final position
            x_dot = np.zeros(3)  # Stop velocity
            x_ddot = np.zeros(3)  # Stop acceleration
            x_dddot = np.zeros(3)  # Stop jerk
            x_ddddot = np.zeros(3)  # Stop snap
        else:
            t_segment = max(0, min(t - t_0, t_end - t_0))
            if segment_idx == 0:
                coefficients = self.coefficients[(segment_idx) * 6: (segment_idx+1) * 6, :]
            else:
                coefficients = self.coefficients[(segment_idx-1) * 6: segment_idx * 6, :]

            T = np.array([t_segment**5, t_segment**4, t_segment**3, t_segment**2, t_segment, 1]).reshape(1, 6)
            T_dot = np.array([5 * t_segment ** 4, 4 * t_segment ** 3,3 * t_segment ** 2,2 * t_segment, 1, 0 ]).reshape(1, 6)
            T_ddot = np.array([20 * t_segment ** 3, 12 * t_segment ** 2, 6 * t_segment, 2, 0, 0]).reshape(1, 6)
            T_dddot = np.array([60 * t_segment ** 2, 24 * t_segment, 6, 0, 0, 0 ]).reshape(1, 6)
            T_ddddot = np.array([120 * t_segment, 24, 0, 0, 0, 0]).reshape(1, 6)

            # Compute x, x_dot, x_ddot, x_dddot, x_ddddot
            x = np.dot(T, coefficients).flatten()
            x_dot = np.dot(T_dot, coefficients).flatten()
            x_ddot = np.dot(T_ddot, coefficients).flatten()
            x_dddot = np.dot(T_dddot, coefficients).flatten()
            x_ddddot = np.dot(T_ddddot, coefficients).flatten()

        # Yaw remains simple
        yaw, yaw_dot = 0, 0

        flat_output = {
            'x': x,
            'x_dot': x_dot,
            'x_ddot': x_ddot,
            'x_dddot': x_dddot,
            'x_ddddot': x_ddddot,
            'yaw': yaw,
            'yaw_dot': yaw_dot
        }
        return flat_output