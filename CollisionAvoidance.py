"""
CollisionAvoidance.py: Collision Detection and Resolution for Wire Arc Additive Manufacturing

This module implements a comprehensive collision detection and avoidance system
for Wire Arc Additive Manufacturing (WAAM) processes. It provides tools for:

1. Collision Detection:
   - Generation of collision candidate points around deposited material
   - Efficient spatial filtering for quick collision checks
   - Exhaustive collision detection for validation purposes

2. Tool Path Optimization:
   - Automatic adjustment of tool orientations to avoid collisions
   - Two-phase optimization algorithm for angle corrections
   - Verification of collision-free solutions

3. Components:
   - CollisionCandidatesGenerator: Manages potential collision points
   - Tool: Simplified geometric model of the deposition tool
   - CollisionAvoidance: Main class coordinating detection and resolution

The system supports both quick optimization-based detection and thorough
exhaustive detection modes. It uses a cylinder-based tool model and implements
vectorized operations for computational efficiency.

Usage:
    collision_manager = CollisionAvoidance(
        trajectory_path="path/to/trajectory.csv",
        bead_width=3.0,     # Width of deposited material (mm)
        bead_height=1.95,   # Height of deposited material (mm)
        tool_radius=12.5,   # Radius of tool body (mm)
        tool_length=17.0    # Length of tool nozzle (mm)
    )

    # Process trajectory and get optimized tool vectors
    new_tool_vectors = collision_manager.process_trajectory()
"""

import numpy as np
import time
from TrajectoryDataManager import TrajectoryManager


# -------------------------------------------------------------------
# CollisionCandidatesGenerator Class
# -------------------------------------------------------------------

class CollisionCandidatesGenerator:
    """
    Handles the generation and management of collision candidate points during additive manufacturing.
    Generates and stores points that could potentially cause collisions with the tool.
    """

    def __init__(self):
        # Arrays to store collision points in 3D space
        self.left_points = None  # Points on the left side of the deposition path (Nx3 array)
        self.right_points = None  # Points on the right side of the deposition path (Nx3 array)
        self.top_points = None  # Points on top of the deposited material (Mx3 array)

        # Layer management
        self.layer_indices = []  # Starting index of each layer
        self.current_layer = -1  # Index of the current layer being processed

        # Bead geometry parameters
        self.bead_width = None  # Width of the deposited bead
        self.bead_height = None  # Height of the deposited bead

        # Track which layers have been processed to avoid redundant generation
        self.already_generated = {}

    def initialize_parameters(self, bead_width, bead_height):
        """
        Initialize the geometric parameters of the deposited bead.

        Args:
            bead_width (float): Width of the deposited bead
            bead_height (float): Height of the deposited bead
        """
        self.bead_width = bead_width
        self.bead_height = bead_height

    def generate_collision_candidates_per_layer(self, points, normal_vectors, build_vectors, layer_index):
        """
        Generate collision candidate points for a specific layer.
        Creates left, right, and top points based on the deposition path and bead geometry.

        Args:
            points (np.array): Points along the deposition path
            normal_vectors (np.array): Normal vectors at each point
            build_vectors (np.array): Build direction vectors at each point
            layer_index (int): Index of the current layer
            is_last_layer (bool): Flag indicating if this is the final layer
        """
        # Skip already processed layers
        if layer_index in self.already_generated:
            return

        # Calculate points on the left and right sides of the bead
        left_points = points + (self.bead_width / 2) * normal_vectors
        right_points = points - (self.bead_width / 2) * normal_vectors

        # Calculate points on top of the bead
        top_points = points + self.bead_height * build_vectors

        # Store or update collision candidates
        if self.left_points is None:
            # Initialize arrays for first layer
            self.left_points = left_points
            self.right_points = right_points
            self.top_points = top_points
        else:
            # Stack new points on existing arrays
            self.left_points = np.vstack((self.left_points, left_points))
            self.right_points = np.vstack((self.right_points, right_points))
            self.top_points = np.vstack((self.top_points, top_points))

        # Update layer tracking information
        self.layer_indices.append(len(self.left_points) - len(points))
        self.current_layer = layer_index
        self.already_generated[layer_index] = True

    def get_all_current_collision_candidates(self):
        """
        Get all currently stored collision candidate points.

        Returns:
            np.array: Combined array of all collision candidate points
        """
        return np.vstack((self.left_points, self.right_points, self.top_points))

    def get_layer_collision_candidates(self, layer_start_idx, layer_end_idx):
        """
        Get collision candidate points for a specific layer range.

        Args:
            layer_start_idx (int): Starting index of the layer range
            layer_end_idx (int): Ending index of the layer range

        Returns:
            tuple: (left points, right points, top points) for the specified layer range
        """
        left = self.left_points[layer_start_idx:layer_end_idx]
        right = self.right_points[layer_start_idx:layer_end_idx]
        top = self.top_points[layer_start_idx:layer_end_idx]
        return left, right, top

# -------------------------------------------------------------------
# Tool Class
# -------------------------------------------------------------------

class Tool:
    """
    Represents a simplified model of the deposition tool for collision detection.
    Models the tool as a cylinder with a specific radius and length.

    This simplified model is sufficient for most collision detection scenarios
    while maintaining computational efficiency.
    """

    def __init__(self, radius: float, nozzle_length: float):
        """
        Initialize the tool model with its geometric parameters.

        Args:
            radius (float): Radius of the tool cylinder (mm)
            nozzle_length (float): Length of the tool nozzle from reference point (mm)
        """
        self.radius = radius  # Tool cylinder radius
        self.nozzle_offset = nozzle_length  # Distance from tool tip to cylinder start

    def get_cylinder_start(self, position: np.ndarray, tool_direction: np.ndarray) -> np.ndarray:
        """
        Calculate the starting point of the tool's cylindrical section.

        The cylinder starts after the nozzle offset, which is important for accurate
        collision detection as the nozzle tip itself is typically narrower.

        Args:
            position (np.ndarray): Current position of the tool tip (3D point)
            tool_direction (np.ndarray): Current orientation vector of the tool

        Returns:
            np.ndarray: 3D coordinates of the cylinder's starting point
        """
        return position + self.nozzle_offset * tool_direction


# -------------------------------------------------------------------
# CollisionAvoidance Class
# -------------------------------------------------------------------

class CollisionAvoidance:
    """
    Main class for detecting and resolving collisions between the deposition tool and
    previously deposited material in additive manufacturing.

    This class implements both an optimized and exhaustive collision detection algorithm,
    along with methods for resolving detected collisions through tool orientation adjustment.
    """

    def __init__(self, trajectory_path, bead_width, bead_height, tool_radius, tool_length, nb_previous_layers=5):
        """
        Initialize collision avoidance system with process parameters.

        Args:
            trajectory_path (str): Path to the trajectory file
            bead_width (float): Width of the deposited material bead
            bead_height (float): Height of the deposited material bead
            tool_radius (float): Radius of the deposition tool
            tool_length (float): Length of the tool nozzle
            nb_previous_layers (int, optional): Number of previous layers to check for collisions. Defaults to 5.
        """
        # Initialize trajectory manager and compute local coordinate bases
        self.trajectory = TrajectoryManager(trajectory_path)
        self.trajectory.compute_and_store_local_bases()

        # Setup collision candidate generator with bead geometry
        self.collision_candidates_generator = CollisionCandidatesGenerator()
        self.collision_candidates_generator.initialize_parameters(bead_width, bead_height)

        # Initialize tool model
        self.tool = Tool(radius=tool_radius, nozzle_length=tool_length)

        # Collision detection storage
        self.collision_points_dict = {}  # Maps point indices to their collision points
        self.all_candidates_dict = {}  # Maps point indices to all nearby points
        self._collision_points = None  # Boolean array marking collision points

        # Metrics storage
        self.problematic_points = []  # List of problematic point indices
        self.tilt_angles = []  # List of correction angles applied

    def get_collision_indices(self):
        """
        Get indices of trajectory points that are in collision.

        Returns:
            np.array: Array of indices where collisions occur
        """
        if self._collision_points is None:
            # Run collision detection if not already done
            self.detect_collisions_optimized()

        return np.where(self._collision_points)[0]

    def detect_collisions_optimized(self):
        """
        Optimized collision detection using spatial filtering.

        This method implements a two-phase collision detection:
        1. Quick filtering using bounding cylinders
        2. Precise collision checking for filtered candidates

        Returns:
            np.array: Boolean array marking points with collisions
        """
        if self._collision_points is not None:
            return self._collision_points

        # Compute average build direction vectors for efficiency
        avg_build_vectors = self.trajectory.compute_average_build_vectors()
        avg_build_vectors /= np.linalg.norm(avg_build_vectors, axis=1)[:, np.newaxis]

        # Pre-compute layer ranges for efficient access
        layer_ranges = [(self.trajectory.layer_indices[i],
                         self.trajectory.layer_indices[i + 1] if i + 1 < len(self.trajectory.layer_indices)
                         else len(self.trajectory.points))
                        for i in range(len(self.trajectory.layer_indices))]

        # Generate collision candidates for all layers
        for layer_idx in range(len(self.trajectory.layer_indices)):
            layer_points = self.trajectory.get_layer_points(layer_idx)
            self.collision_candidates_generator.generate_collision_candidates_per_layer(
                points=layer_points,
                normal_vectors=self.trajectory.n_vectors[layer_idx],
                build_vectors=self.trajectory.b_vectors[layer_idx],
                layer_index=layer_idx,
            )

        collision_points = np.zeros(len(self.trajectory.points), dtype=bool)
        self.collision_points_dict = {}
        self.all_candidates_dict = {}

        # Pre-allocate arrays for vector calculations
        max_points = max(len(range(*r)) for r in layer_ranges)
        diff_vectors = np.empty((max_points * 2, 3))  # Space for left and right points

        # Main collision detection loop
        for point_idx, current_point in enumerate(self.trajectory.points):
            current_layer = np.searchsorted(self.trajectory.layer_indices[1:], point_idx)

            collision_mask = None
            all_points = None

            # Check each layer up to current layer
            for layer in range(current_layer + 1):
                start_idx, end_idx = layer_ranges[layer]
                if layer == current_layer:
                    end_idx = min(end_idx, point_idx)

                # Get candidate points for current layer
                left, right, top = self.collision_candidates_generator.get_layer_collision_candidates(
                    start_idx, end_idx)

                if len(left) > 0 or len(right) > 0:
                    # Efficient point concatenation
                    test_points = np.vstack([p for p in [left, right] if len(p) > 0])

                    # Vectorized projection calculation
                    np.subtract(test_points, current_point, out=diff_vectors[:len(test_points)])
                    projections = np.abs(np.dot(diff_vectors[:len(test_points)],
                                                avg_build_vectors[layer]))

                    # Quick filtering using larger radius
                    point_mask = projections <= self.tool.radius * 1.1

                    if np.any(point_mask):
                        if collision_mask is None:
                            collision_mask = point_mask
                            all_points = test_points
                        else:
                            collision_mask = np.concatenate([collision_mask, point_mask])
                            all_points = np.vstack([all_points, test_points])

                # Special handling for top points of current layer
                if layer == current_layer and len(top) > 0:
                    if collision_mask is None:
                        collision_mask = np.ones(len(top), dtype=bool)
                        all_points = top
                    else:
                        collision_mask = np.concatenate([collision_mask, np.ones(len(top), dtype=bool)])
                        all_points = np.vstack([all_points, top])

            # Final precise collision check
            if collision_mask is not None and np.any(collision_mask):
                filtered_points = all_points[collision_mask]
                distances = self.compute_point_cylinder_distances(
                    filtered_points,
                    self.tool.get_cylinder_start(current_point, self.trajectory.tool_vectors[point_idx]),
                    self.trajectory.tool_vectors[point_idx]
                )

                if np.any(distances < self.tool.radius):
                    collision_points[point_idx] = True
                    self.collision_points_dict[point_idx] = filtered_points[distances < self.tool.radius]
                    self.all_candidates_dict[point_idx] = filtered_points

        self._collision_points = collision_points
        return collision_points

    def detect_collisions_exhaustive(self):
        """
        Perform exhaustive collision detection without spatial filtering.

        This method checks all possible collision points without optimization.
        Useful for validation or when dealing with complex geometries where
        spatial filtering might miss edge cases.

        Returns:
            np.array: Boolean array marking points with collisions
        """
        if self._collision_points is not None:
            return self._collision_points

        collision_points = np.zeros(len(self.trajectory.points), dtype=bool)
        self.collision_points_dict = {}
        self.all_candidates_dict = {}

        # Generate collision candidates for all layers upfront
        for layer_idx in range(len(self.trajectory.layer_indices)):
            layer_points = self.trajectory.get_layer_points(layer_idx)
            self.collision_candidates_generator.generate_collision_candidates_per_layer(
                points=layer_points,
                normal_vectors=self.trajectory.n_vectors[layer_idx],
                build_vectors=self.trajectory.b_vectors[layer_idx],
                layer_index=layer_idx,
            )

        # Check each point against all relevant previous geometry
        for point_idx, current_point in enumerate(self.trajectory.points):
            current_layer = np.searchsorted(self.trajectory.layer_indices[1:], point_idx)
            collision_candidates = []

            # Check against each layer up to current layer
            for layer in range(current_layer + 1):
                start_idx = self.trajectory.layer_indices[layer]
                end_idx = (self.trajectory.layer_indices[layer + 1]
                           if layer + 1 < len(self.trajectory.layer_indices)
                           else len(self.trajectory.points))

                # Adjust end index for current layer
                if layer == current_layer:
                    end_idx = min(end_idx, point_idx)

                # Get all candidate points from current layer
                left_points, right_points, top_points = \
                    self.collision_candidates_generator.get_layer_collision_candidates(start_idx, end_idx)

                # Collect all potential collision points
                if len(left_points) > 0:
                    collision_candidates.append(left_points)
                if len(right_points) > 0:
                    collision_candidates.append(right_points)
                if layer == current_layer and len(top_points) > 0:
                    collision_candidates.append(top_points)

            # Perform collision check if candidates exist
            if collision_candidates:
                all_candidates = np.vstack(collision_candidates)
                distances = self.compute_point_cylinder_distances(
                    all_candidates,
                    self.tool.get_cylinder_start(current_point, self.trajectory.tool_vectors[point_idx]),
                    self.trajectory.tool_vectors[point_idx]
                )

                if np.any(distances < self.tool.radius):
                    collision_points[point_idx] = True
                    self.collision_points_dict[point_idx] = all_candidates[distances < self.tool.radius]
                    self.all_candidates_dict[point_idx] = all_candidates

        self._collision_points = collision_points
        return self._collision_points

    def get_problematic_points(self):
        """
        Get indices of all problematic points in the trajectory.

        Returns:
            list: List of indices where collisions were detected and processed
        """
        return self.problematic_points

    def get_correction_angles(self):
        """
        Get all correction angles that were applied to resolve collisions.

        Returns:
            list: List of correction angles in degrees
        """
        return self.tilt_angles

    @staticmethod
    def compute_point_cylinder_distances(points, cylinder_start, cylinder_axis):
        """
        Calculate distances between points and a semi-infinite cylinder.

        Computes the shortest distance between each point and the cylinder surface
        using vectorized operations for efficiency.

        Args:
            points (np.ndarray): Array of points to check (Nx3)
            cylinder_start (np.ndarray): Starting point of cylinder axis (3,)
            cylinder_axis (np.ndarray): Direction vector of cylinder axis (3,)

        Returns:
            np.ndarray: Array of distances from each point to cylinder surface
        """
        # Calculate vectors from cylinder start to each point
        vectors_to_points = points - cylinder_start

        # Project these vectors onto cylinder axis
        projections = np.dot(vectors_to_points, cylinder_axis)

        # Consider only positive projections (semi-infinite cylinder)
        projections = np.maximum(projections, 0)

        # Find closest points on cylinder axis
        projected_points = cylinder_start + projections[:, np.newaxis] * cylinder_axis

        # Calculate distances from points to their projections
        distances = np.linalg.norm(points - projected_points, axis=1)
        return distances

    def process_trajectory(self):
        """
        Main trajectory processing method for collision avoidance.

        This method performs the following steps:
        1. Detects initial collisions in the trajectory
        2. Attempts to resolve each collision by adjusting tool angles
        3. Verifies the effectiveness of collision resolution
        4. Reports processing metrics and results

        Returns:
            np.ndarray: Updated tool vectors with collision-free orientations
        """
        print("Trajectory processing...")
        start_time = time.time()

        # Use pre-computed basis vectors for efficiency
        t_vec = self.trajectory.t_vectors  # Tangent vectors
        n_vec = self.trajectory.n_vectors  # Normal vectors
        b_vec = self.trajectory.b_vectors  # Build vectors
        tool_vec = self.trajectory.tool_vectors.copy()
        updated_tool_vec = tool_vec.copy()

        # Initial collision detection
        trajectory_points_colliding = self.detect_collisions_optimized()
        # trajectory_points_colliding = self.detect_collisions_exhaustive()

        # Identify problematic points
        problematic_points = np.where(trajectory_points_colliding)[0]
        print(f"\nNumber of initial trajectory points colliding: {len(problematic_points)}")

        if len(problematic_points) == 0:
            print("No collisions to resolve")
            return tool_vec

        # Initialize collision resolution
        total_resolved = 0
        correction_angles = []  # For calculating average correction
        print("\nStarting collision resolution:")
        print("-" * 50)

        # Process each problematic point
        for point_idx in problematic_points:
            if point_idx not in self.collision_points_dict:
                print(f"Point {point_idx}: Skipped (no collision data)")
                continue

            print(f"\nProcessing point {point_idx}:")

            # Calculate initial tool angle
            initial_angle = np.arctan2(
                np.dot(updated_tool_vec[point_idx], b_vec[point_idx]),
                np.dot(updated_tool_vec[point_idx], n_vec[point_idx])
            )

            # Attempt collision resolution
            new_angle = self.calculate_tilt_angle(
                point_idx,
                self.collision_points_dict[point_idx],
                t_vec[point_idx],
                n_vec[point_idx],
                b_vec[point_idx],
                updated_tool_vec[point_idx]
            )

            if new_angle is not None:
                # Update tool vector with new orientation
                updated_tool_vec[point_idx] = (
                        n_vec[point_idx] * np.cos(new_angle) +
                        b_vec[point_idx] * np.sin(new_angle)
                )

                # Verify collision resolution
                distances = self.compute_point_cylinder_distances(
                    self.all_candidates_dict[point_idx],
                    self.tool.get_cylinder_start(
                        self.trajectory.points[point_idx],
                        updated_tool_vec[point_idx]
                    ),
                    updated_tool_vec[point_idx]
                )

                if np.all(distances >= self.tool.radius):
                    total_resolved += 1
                    # Calculate correction magnitude
                    angle_diff = new_angle - initial_angle
                    # Normalize angle difference to [-pi, pi]
                    while angle_diff > np.pi:
                        angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi:
                        angle_diff += 2 * np.pi

                    correction_angles.append(abs(angle_diff))
                    # Store metrics
                    self.problematic_points.append(point_idx)
                    angle_corrige = float(f"{np.degrees(abs(angle_diff)):.2f}")
                    self.tilt_angles.append(angle_corrige)

                    print(f"Point {point_idx}: RESOLVED")
                    print(f"   Correction amplitude: {np.degrees(abs(angle_diff)):.2f}°")
                else:
                    print(f"Point {point_idx}: FAILED (Collision still present after angle correction)")
            else:
                print(f"Point {point_idx}: FAILED (No valid angle found)")

        # Final verification and statistics
        final_collision_count, residual_points = self.verify_collisions(updated_tool_vec)
        print("\n" + "-" * 50)
        print(f"Total points resolved: {len(problematic_points) - final_collision_count}/{len(problematic_points)}")
        print(
            f"Resolution rate: {((len(problematic_points) - final_collision_count) / len(problematic_points) * 100):.1f}%")
        if correction_angles:
            mean_correction = np.degrees(np.mean(correction_angles))
            print(f"Average correction amplitude: {mean_correction:.2f}°")
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")

        print(f"\nFinal verification:")
        print(f"Number of residual collisions: {final_collision_count}")
        print(f"Problematic points: {residual_points}")

        return updated_tool_vec

    def verify_collisions(self, updated_tool_vectors=None):
        """
        Verify residual collisions after tool vector modification.

        Checks all previously colliding points with their updated tool orientations
        to ensure collisions have been properly resolved.

        Args:
            updated_tool_vectors (np.ndarray, optional): Modified tool orientation vectors.
                If None, uses original trajectory vectors.

        Returns:
            tuple: (Number of remaining collisions, List of points still in collision)
        """
        tool_vectors = updated_tool_vectors if updated_tool_vectors is not None else self.trajectory.tool_vectors
        collision_count = 0
        residual_collisions = []

        # Check only previously colliding points for efficiency
        initial_collision_points = np.where(self._collision_points)[0]

        for point_idx in initial_collision_points:
            if point_idx in self.collision_points_dict:
                distances = self.compute_point_cylinder_distances(
                    self.collision_points_dict[point_idx],
                    self.tool.get_cylinder_start(
                        self.trajectory.points[point_idx],
                        tool_vectors[point_idx]
                    ),
                    tool_vectors[point_idx]
                )
                if np.any(distances < self.tool.radius):
                    collision_count += 1
                    residual_collisions.append(point_idx)

        return collision_count, residual_collisions

    # ----------------------------------------------------------------------------------
    # Algorithm for tool angle modification
    # ----------------------------------------------------------------------------------

    def calculate_tilt_angle(self, point_idx, collision_points, t, n, b, tool_vec):
        """
        Calculate the optimal tool tilt angle to avoid collisions using a two-phase approach.

        This method implements a sophisticated algorithm that:
        1. Determines the best direction to tilt the tool by analyzing the distribution
           of collision points in different quadrants
        2. Performs a coarse search with predefined angles to find an initial valid solution
        3. Refines the solution by gradually reducing the tilt angle while maintaining
           collision-free status

        Args:
            point_idx (int): Index of the current trajectory point
            collision_points (np.ndarray): Points causing collisions with the tool
            t (np.ndarray): Tangent vector at the current point
            n (np.ndarray): Normal vector at the current point
            b (np.ndarray): Build direction vector at the current point
            tool_vec (np.ndarray): Current tool orientation vector

        Returns:
            float or None: Optimal tilt angle in radians, or None if no valid solution found
        """
        # Get current point and all nearby points for collision checking
        current_point = self.trajectory.points[point_idx]
        all_candidate_points = self.all_candidates_dict[point_idx]

        # Calculate current tool angle in the n-b plane
        # arctan2 gives angle in [-π, π] range
        current_angle = np.arctan2(np.dot(tool_vec, b), np.dot(tool_vec, n))

        # Project collision points onto the n-b plane for quadrant analysis
        # First center points relative to current position
        centered_points = collision_points - current_point
        # Then project onto n-b coordinate system
        projected_points = np.column_stack((
            np.dot(centered_points, n),  # x coordinates in n-b plane
            np.dot(centered_points, b)  # y coordinates in n-b plane
        ))

        # Analyze distribution of collision points in quadrants
        # Calculate angles of collision points relative to current tool orientation
        angle_diffs = np.arctan2(projected_points[:, 1], projected_points[:, 0]) - current_angle
        # Normalize angles to [-π, π] range for consistent quadrant analysis
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

        # Count points in critical quadrants
        quad1_count = np.sum((0 < angle_diffs) & (angle_diffs < np.pi / 2))  # Upper right
        quad2_count = np.sum((-np.pi / 2 < angle_diffs) & (angle_diffs < 0))  # Lower right

        # Determine tilt direction based on quadrant with fewer collision points
        direction = 1 if quad2_count > quad1_count else -1

        # Phase 1: Coarse Search
        # Try increasingly larger angles until finding collision-free orientation
        coarse_angles = np.radians([1, 15, 30, 45, 60, 90])  # Progressive angle steps
        first_solution = None

        for angle in coarse_angles:
            test_angle = current_angle + direction * angle
            # Calculate new tool vector in n-b plane using rotation matrix elements
            test_tool_vec = n * np.cos(test_angle) + b * np.sin(test_angle)

            # Check if this orientation resolves all collisions
            distances = self.compute_point_cylinder_distances(
                all_candidate_points,
                self.tool.get_cylinder_start(current_point, test_tool_vec),
                test_tool_vec
            )

            if np.all(distances >= self.tool.radius):
                first_solution = test_angle
                break

        # If no solution found in coarse search, return None
        if first_solution is None:
            return None

        # Phase 2: Fine Optimization
        # Gradually reduce the tilt angle while maintaining collision-free status
        fine_step = np.radians(1.0)  # Initial step size (1 degree)
        current_test_angle = first_solution
        best_valid_angle = first_solution

        # Continue optimization until step size becomes very small
        while fine_step > np.radians(0.1):  # Stop at 0.1 degree precision
            # Try to reduce angle while maintaining collision-free status
            test_angle = current_test_angle - direction * fine_step
            test_tool_vec = n * np.cos(test_angle) + b * np.sin(test_angle)

            # Check if reduced angle maintains collision-free status
            distances = self.compute_point_cylinder_distances(
                all_candidate_points,
                self.tool.get_cylinder_start(current_point, test_tool_vec),
                test_tool_vec
            )

            if np.all(distances >= self.tool.radius):
                # Update best solution if still collision-free
                best_valid_angle = test_angle
                current_test_angle = test_angle
            else:
                # If collision occurs, reduce step size and try again
                fine_step /= 2

        return best_valid_angle