from heapq import heappush, heappop  # Recommended.
import numpy as np
from math import dist
from flightsim.world import World

from occupancy_map import OccupancyMap # Recommended.

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    def dijkstra(world_d, resolution_d, margin_d, start_d, goal_d):
        """
        Parameters:
            world_d,      World object representing the environment obstacles
            resolution_d, xyz resolution in meters for an occupancy map, shape=(3,)
            margin_d,     minimum allowed distance in meters from path to obstacles.
            start_d,      xyz position in meters, shape=(3,)
            goal_d,       xyz position in meters, shape=(3,)
        Output:
            return a tuple (path, nodes_expanded)
            path,       xyz position coordinates along the path in meters with
                        shape=(N,3). These are typically the centers of visited
                        voxels of an occupancy map. The first point must be the
                        start and the last point must be the goal. If no path
                        exists, return None.
            nodes_expanded, the number of nodes that have been expanded
        """
        # 1. Initialization
        occ_map = OccupancyMap(world_d, resolution_d, margin_d)  # Initialize the occupancy map
        start_index = tuple(occ_map.metric_to_index(start_d))  # Convert start position to voxel index
        goal_index = tuple(occ_map.metric_to_index(goal_d))  # Convert goal position to voxel index

        # Priority queue: stores (cost, node)
        frontier = []
        heappush(frontier, (0, start_index))

        # Cost dictionary: shortest known cost to reach each node
        cost_so_far = {start_index: 0}
        # Path dictionary: stores parent of each node to reconstruct the path
        came_from = {start_index: None}

        nodes_expanded = 0

        # 2. Graph Search
        while frontier:
            current_cost, current_node = heappop(frontier)  # Get node with lowest cost
            nodes_expanded += 1

            # Goal check
            if current_node == goal_index:
                break

            # Expand neighbors
            for neighbor in occ_map.find_neighbor_index(current_node):
                # Convert neighbor to tuple for consistency
                neighbor_key = tuple(neighbor)

                # Calculate new cost
                new_cost = current_cost + dist(neighbor, current_node)

                # If new path is better or neighbor is not visited yet
                if neighbor_key not in cost_so_far or new_cost < cost_so_far[neighbor_key]:
                    cost_so_far[neighbor_key] = new_cost
                    heappush(frontier, (new_cost, neighbor_key))
                    came_from[neighbor_key] = current_node

        # 3. Reconstruct Path
        if goal_index not in came_from:
            return None, nodes_expanded  # No path found
        path = [goal_d]
        node = came_from[goal_index]
        while node != start_index:
            path.append(occ_map.index_to_metric_center(node))
            node = came_from[node]
        path.append(start_d)
        path.reverse()
        return np.array(path), nodes_expanded

    def a_star(world_a, resolution_a, margin_a, start_a, goal_a):
        """
        Parameters:
            world_a,      World object representing the environment obstacles
            resolution_a, xyz resolution in meters for an occupancy map, shape=(3,)
            margin_a,     minimum allowed distance in meters from path to obstacles.
            start_a,      xyz position in meters, shape=(3,)
            goal_a,       xyz position in meters, shape=(3,)
        Output:
            return a tuple (path, nodes_expanded)
            path,       xyz position coordinates along the path in meters with
                        shape=(N,3). These are typically the centers of visited
                        voxels of an occupancy map. The first point must be the
                        start and the last point must be the goal. If no path
                        exists, return None.
            nodes_expanded, the number of nodes that have been expanded
        """
        # 1. Initialization
        occ_map = OccupancyMap(world_a, resolution_a, margin_a)
        start_index = tuple(occ_map.metric_to_index(start_a))
        goal_index = tuple(occ_map.metric_to_index(goal_a))

        frontier = []
        heappush(frontier, (0, start_index))

        cost_so_far = {start_index: 0}
        came_from = {start_index: None}

        nodes_expanded = 0

        # 2. Graph Search
        while frontier:
            current_cost, current_node = heappop(frontier)
            nodes_expanded += 1

            if current_node == goal_index:
                break

            for neighbor in occ_map.find_neighbor_index(current_node):
                neighbor_key = tuple(neighbor)
                step_cost = dist(neighbor, current_node)
                new_cost = cost_so_far[current_node] + step_cost

                if neighbor_key not in cost_so_far or new_cost < cost_so_far[neighbor_key]:
                    cost_so_far[neighbor_key] = new_cost
                    priority = new_cost + dist(neighbor_key, goal_index)
                    heappush(frontier, (priority, neighbor_key))
                    came_from[neighbor_key] = current_node

        # 3. Reconstruct Path
        if goal_index not in came_from:
            return None, nodes_expanded
        path = [goal_a]
        node = came_from[goal_index]
        while node != start_index:
            path.append(occ_map.index_to_metric_center(node))
            node = came_from[node]
        path.append(start_a)
        path.reverse()
        return np.array(path), nodes_expanded

    if astar:
        return a_star(world, resolution, margin, start, goal)
    else:
        return dijkstra(world, resolution, margin, start, goal)
