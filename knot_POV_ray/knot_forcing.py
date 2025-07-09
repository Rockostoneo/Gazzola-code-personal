import numpy as np
from numba import njit
from elastica.typing import SystemType
from elastica.external_forces import NoForces


class MultiTargetForce(NoForces):
    """
    This class applies directional forces on the end node towards a sequence of targets.

        Attributes
        ----------
        force_mag: float
            Magnitude of the force applied to the end node.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        targets: numpy.ndarray
            2D (dim) array containing data with 'float' type. Target positions for the end node.

    """

    def __init__(self, force_mag, ramp_up_time, targets):
        """

        Parameters
        ----------
        force_mag: float
            Magnitude of the force applied to the end node.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        targets: numpy.ndarray
            2D (dim) array containing data with 'float' type. Target positions for the end node.

        """
        super(MultiTargetForce, self).__init__()
        self.force_mag = force_mag
        assert ramp_up_time > 0.0
        self.ramp_up_time = ramp_up_time
        self.targets = targets
        self.current_target = 0
        self.last_target_time = 0.0


    def apply_forces(self, system: SystemType, time=0.0):
        if (np.linalg.norm(self.targets[self.current_target] - system.position_collection[..., -1]) < .04) & (self.current_target < len(self.targets) - 1): 
            self.current_target += 1
            self.last_target_time = time
        self.compute_end_point_forces(
            system.external_forces,
            self.force_mag,
            time,
            self.ramp_up_time,
            self.targets,
            system.position_collection,
            self.current_target,
            self.last_target_time
        )
        

    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces(
        external_forces, force_mag, time, ramp_up_time, targets, node_positions, current_target,last_target_time
    ):
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        time: float
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        end_force = force_mag * (targets[current_target] - node_positions[..., -1]) / np.linalg.norm(targets[current_target] - node_positions[..., -1])
        time_factor = min(1.0, (time - last_target_time) / ramp_up_time) +0.3   #ramp up force
        distance_factor = min(np.linalg.norm(targets[current_target] - node_positions[..., -1]), 1) * 0.5 + 0.5  #reduce force with distance to target
        external_forces[..., -1] += end_force * time_factor * distance_factor


# TODO Clean up docstrings 
class SnapForce(NoForces):
    """
    This class applies directional forces on the end node.

        Attributes
        ----------
        force_mag: float
            Magnitude of the force applied to the end node.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        distance: float
            Distance to the target at which the node will start snapping
        snap_nodes: numpy.ndarray
            1D (dim) array containing data with 'int' type. Node indices that are snapped to targets.
        snap_targets: numpy.ndarray
            2D (dim) array containing data with 'float' type. Target positions for the snapped nodes.
        stop_target: numpy.ndarray
            1D (dim) array containing data with 'float' type. Target position for the end node to stop all snapping.
        stop_distance: float
            Distance to the stop target at which snapping stops.

    """

    def __init__(self, force_mag, snap_nodes, snap_targets, ramp_up_time=1e-2, distance = 0.01,stop_target=None,stop_distance=0.01):
        """
        Parameters      
        ----------
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        """
        super(SnapForce, self).__init__()
        self.force_mag = force_mag
        self.ramp_up_time = ramp_up_time
        self.snap_nodes = snap_nodes
        self.snap_targets = snap_targets
        self.snap_time = np.zeros(snap_nodes.shape[0])
        self.distance = distance
        self.stop_target = stop_target
        self.stop_distance = stop_distance
        self.active = True


    def apply_forces(self, system: SystemType, time=0.0):
        if self.stop_target is not None and np.linalg.norm(system.position_collection[..., -1] - self.stop_target) < self.stop_distance:
            self.active = False
        if not self.active:
            return    
        for i in range(len(self.snap_time)):
            if (self.snap_time[i] == 0) and (np.linalg.norm(self.snap_targets[i] - system.position_collection[..., self.snap_nodes[i]]) < self.distance):
                self.snap_time[i] = time
        self.compute_end_point_forces(
            system.external_forces,
            self.force_mag,
            time,
            self.ramp_up_time,
            system.position_collection,
            self.snap_nodes,
            self.snap_targets,
            self.snap_time
        )
        

    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces(
        external_forces, force_mag, time, ramp_up_time, position_collection, snap_nodes, snap_targets,last_snap_time
    ):
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        force_mag: float
            Magnitude of the force applied to the end node.
        time: float
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        position_collection: numpy.ndarray
            Array containing data with 'float' type. Position of the nodes.
        snap_nodes: numpy.ndarray
            1D (dim) array containing data with 'int' type. Node indices that are snapped to targets.
        snap_targets: numpy.ndarray
            2D (dim) array containing data with 'float' type. Target positions for the snapped nodes.
        snap_time: numpy.ndarray
            1D (dim) array containing data with 'float' type. Time the node started snapping.
        """
        for i in range(len(snap_nodes)):
            if (last_snap_time[i] != 0):
                factor = min(1.0, ((time - last_snap_time[i]) / ramp_up_time)**2)
                snap_force = force_mag * (snap_targets[i] - position_collection[..., snap_nodes[i]]) / np.linalg.norm(snap_targets[i] - position_collection[..., snap_nodes[i]])
                distance_factor = min(np.linalg.norm(snap_targets[i] - position_collection[..., snap_nodes[i]]), 1) * 0.2 + 0.8
                external_forces[..., snap_nodes[i]] += snap_force * factor * distance_factor
            


