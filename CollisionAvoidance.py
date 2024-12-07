import numpy as np
import time
from TrajectoryDataManager import TrajectoryManager

# -------------------------------------------------------------------
# Collision candidates class
# -------------------------------------------------------------------

class CollisionCandidatesGenerator :
    def __init__(self):
        # Numpy array structure to stack collision candidates
        self.left_points = None  # Array Nx3
        self.right_points = None  # Array Nx3
        self.top_points = None  # Array Mx3

        # Layer info
        self.layer_indices = []  # starting index
        self.current_layer = -1  # current layer

        # Geometrical params
        self.w_layer = None  # bead width
        self.h_layer = None  # bead height

        self.already_generated = {}
    def initialize_parameters(self, w_layer, h_layer):
        """Bead parameters initialization"""
        self.w_layer = w_layer
        self.h_layer = h_layer

    def generate_collision_candidates_per_layer(self, points, normal_vectors, build_vectors, layer_index, is_last_layer):
        """Generating collision candidates for a layer"""
        # Checking if the layer was already treated
        if layer_index in self.already_generated:
            return

        # Calculating left and right points
        left_points = points + (self.w_layer / 2) * normal_vectors
        right_points = points - (self.w_layer / 2) * normal_vectors

        # Update collision candidates
        if self.left_points is None:
            self.left_points = left_points
            self.right_points = right_points
        else:
            self.left_points = np.vstack((self.left_points, left_points))
            self.right_points = np.vstack((self.right_points, right_points))

        # Last layer exception
        if is_last_layer:
            top_points = points + self.h_layer * build_vectors
            self.top_points = top_points

        # Updating layer indices
        self.layer_indices.append(len(self.left_points) - len(points))
        self.current_layer = layer_index

        # Mark the layer to treated
        self.already_generated[layer_index] = True

    def get_all_current_collision_candidates(self):
        """ Return current collision candidates """
        points_list = [self.left_points, self.right_points]
        if self.top_points is not None:
            points_list.append(self.top_points)
        return np.vstack(points_list)

# -------------------------------------------------------------------
# Tool class
# -------------------------------------------------------------------

class Tool:
    def __init__(self, radius: float, nozzle_length: float):
        """ Simplified tool for collision detection """
        self.radius = radius
        self.nozzle_offset = nozzle_length

    def get_cylinder_start(self, position: np.ndarray, tool_direction: np.ndarray):
        """ Calculate starting point of the cylinder (after the nozzle) by using tool_direction """
        return position + self.nozzle_offset * tool_direction

# -------------------------------------------------------------------
# CollisionAvoidance class
# -------------------------------------------------------------------

class CollisionAvoidance:
    def __init__(self, trajectory_path, bead_width, bead_height, tool_radius, tool_length):
        # Create trajectory manager instance
        self.trajectory = TrajectoryManager(trajectory_path)

        # Rest of the initialization remains the same
        self.collision_candidates_generator = CollisionCandidatesGenerator()
        self.collision_candidates_generator.initialize_parameters(bead_width, bead_height)

        self.tool = Tool(radius=tool_radius, nozzle_length=tool_length)

        self.collision_candidates_dict = {}
        self.collision_points_dict = {}

        self._initial_collisions_detected = False
        self._collision_points = None

    def update_current_collision_candidates_list(self, point_idx, layer):
        """ Update the list of collision candidates """
        # If layer change
        if layer != self.current_layer:
            # Remove previous top points
            if self.collision_candidates_generator.top_points is not None:
                # Find the index of the starting top point of the previous layer
                layer_start = self.collision_candidates_generator.layer_indices[self.current_layer]
                # If last layer use trajectory len
                if self.current_layer + 1 < len(self.collision_candidates_generator.layer_indices):
                    layer_end = self.collision_candidates_generator.layer_indices[self.current_layer + 1]
                else:
                    layer_end = len(self.collision_candidates_generator.left_points)

                num_points = layer_end - layer_start
                # Remove the top points collision candidates of the previous last layer
                self.current_collision_candidates_list = self.current_collision_candidates_list[:-num_points]

            self.current_layer = layer

        # Add new left and right points
        left_point = self.collision_candidates_generator.left_points[point_idx:point_idx + 1]
        right_point = self.collision_candidates_generator.right_points[point_idx:point_idx + 1]

        new_points = np.vstack([left_point, right_point])

        # Add top points if new last layer
        if layer == len(self.trajectory.layer_indices) - 1:
            top_point = self.collision_candidates_generator.top_points[point_idx:point_idx + 1]
            new_points = np.vstack([new_points, top_point])

        # Updating the current collision candidates list
        if len(self.current_collision_candidates_list) == 0:
            self.current_collision_candidates_list = new_points
        else:
            self.current_collision_candidates_list = np.vstack([self.current_collision_candidates_list, new_points])

        return self.current_collision_candidates_list

    def check_collisions(self, tool_vec):
        """" Find initial trajectory points creating a collision between the tool and the collision candidates"""
        collisions_detected = np.zeros(len(self.trajectory.points), dtype=bool)
        self.current_collision_candidates_list = np.array([])
        self.current_layer = 0
        self.collision_points_dict = {}
        self.collision_candidates_dict = {}

        for layer_idx in range(len(self.trajectory.layer_indices)):
            start_idx = self.trajectory.layer_indices[layer_idx]
            end_idx = self.trajectory.layer_indices[layer_idx + 1] if layer_idx + 1 < len(
                self.trajectory.layer_indices) else len(self.trajectory.points)

            for i in range(end_idx - start_idx):
                point_idx = start_idx + i
                points_to_check = self.update_current_collision_candidates_list(point_idx, layer_idx)

                if len(points_to_check) > 0:
                    cylinder_start = self.tool.get_cylinder_start(
                        self.trajectory.points[point_idx],
                        tool_vec[point_idx]
                    )
                    distances = self.compute_point_cylinder_distances(
                        points_to_check,
                        cylinder_start,
                        tool_vec[point_idx]
                    )
                    is_collision = np.any(distances < self.tool.radius)
                    collisions_detected[point_idx] = is_collision

                    if is_collision:
                        self.collision_candidates_dict[point_idx] = points_to_check.copy()
                        collision_indices = np.where(distances < self.tool.radius)[0]
                        self.collision_points_dict[point_idx] = points_to_check[collision_indices]

        return collisions_detected

    def detect_initial_collisions(self):
        """Launch initial detection collision"""
        if self._initial_collisions_detected:
            return self._collision_points

        t_vec, n_vec, b_vec, tool_vec = self.trajectory.calculate_local_basis()

        for layer_idx in range(len(self.trajectory.layer_indices)):
            layer_points = self.trajectory.get_layer_points(layer_idx)
            self.collision_candidates_generator.generate_collision_candidates_per_layer(
                points=layer_points,
                normal_vectors=n_vec[layer_idx],
                build_vectors=b_vec[layer_idx],
                layer_index=layer_idx,
                is_last_layer=(layer_idx == len(self.trajectory.layer_indices) - 1)
            )

        self._collision_points = self.check_collisions(tool_vec)
        self._initial_collisions_detected = True

        return self._collision_points

    @staticmethod
    def compute_point_cylinder_distances(points, cylinder_start, cylinder_axis):
        """ Vectorized distances calculus between points and cylinder """
        # Vector between collision candidates and cylinder start
        vectors_to_points = points - cylinder_start

        # Projecting on cylinder axis
        projections = np.dot(vectors_to_points, cylinder_axis)

        # For semi-infinite cylinder we only keep positive projections
        projections = np.maximum(projections, 0)

        projected_points = cylinder_start + projections[:, np.newaxis] * cylinder_axis

        # Distances between initial points and there projection on cylinder axis
        distances = np.linalg.norm(points - projected_points, axis=1)
        return distances

    def process_trajectory(self):
        """Main function for collision avoidance processing"""
        start_time = time.time()
        print("Trajectory processing...")

        # Vector acquisition
        t_vec, n_vec, b_vec, tool_vec = self.trajectory.calculate_local_basis()

        # Use existing collision detection results if available
        if not self._initial_collisions_detected:
            print("\nCollision candidates generation...")
            for layer_idx in range(len(self.trajectory.layer_indices)):
                layer_points = self.trajectory.get_layer_points(layer_idx)
                self.collision_candidates_generator.generate_collision_candidates_per_layer(
                    points=layer_points,
                    normal_vectors=n_vec[layer_idx],
                    build_vectors=b_vec[layer_idx],
                    layer_index=layer_idx,
                    is_last_layer=(layer_idx == len(self.trajectory.layer_indices) - 1)
                )

            generation_time = time.time() - start_time
            print(f"\nGeneration execution time: {generation_time:.2f} seconds")

            print("\nInitial collision detection...")
            trajectory_points_colliding = self.check_collisions(tool_vec)
        else:
            print("\nUsing existing collision detection results...")
            trajectory_points_colliding = self._collision_points

        problematic_points = np.where(trajectory_points_colliding)[0]
        print(f"\nNumber of initial trajectory points colliding: {len(problematic_points)}")

        if len(problematic_points) == 0:
            print("No collisions to resolve")
            return tool_vec

        # Iterative collision resolution
        print("\nStarting iterative collision resolution...")
        updated_tool_vec = tool_vec.copy()
        total_resolved = 0

        for point_idx in problematic_points:
            if point_idx not in self.collision_points_dict:
                continue

            # Get all candidates for this point for complete collision checking
            all_candidates = self.collision_candidates_dict[point_idx]
            collision_points = self.collision_points_dict[point_idx]

            # Calculate new tilt angle using the new method
            new_angle = self.calculate_tilt_angle(
                point_idx,
                collision_points,
                all_candidates,
                t_vec[point_idx],
                n_vec[point_idx],
                updated_tool_vec[point_idx]
            )

            if new_angle is not None:
                # Update tool vector with new angle
                updated_tool_vec[point_idx] = (
                        t_vec[point_idx] * np.cos(new_angle) +
                        n_vec[point_idx] * np.sin(new_angle)
                )

                # Verify the new orientation resolves all collisions
                cylinder_start = self.tool.get_cylinder_start(
                    self.trajectory.points[point_idx],
                    updated_tool_vec[point_idx]
                )

                distances = self.compute_point_cylinder_distances(
                    all_candidates,
                    cylinder_start,
                    updated_tool_vec[point_idx]
                )

                initial_angle = np.arctan2(np.dot(tool_vec[point_idx], n_vec[point_idx]),
                                           np.dot(tool_vec[point_idx], t_vec[point_idx]))
                angle_sector = new_angle - initial_angle

                while angle_sector > np.pi:
                    angle_sector -= 2 * np.pi
                while angle_sector < -np.pi:
                    angle_sector += 2 * np.pi

                if not np.any(distances < self.tool.radius):
                    total_resolved += 1
                    print(
                        f"Point {point_idx} resolved with angle {np.degrees(new_angle):.2f}° (rotation sector: {np.degrees(angle_sector):.2f}°)")
                else:
                    print(f"Warning: Could not fully resolve collisions for point {point_idx}")
            else:
                print(f"Warning: No valid angle found for point {point_idx}")

        resolution_time = time.time() - start_time
        print(f"\nTotal points resolved: {total_resolved}/{len(problematic_points)}")
        print(f"Total execution time: {resolution_time:.2f} seconds")

        return updated_tool_vec

    def calculate_tilt_angle(self, point_idx, collision_points, all_candidate_points, t, n, tool_vec):
        """
        Calculate the minimal angle needed to avoid all collisions.
        """
        current_trajectory_point = self.trajectory.points[point_idx]

        # Project only collision points to determine rotation direction
        centered_collisions = collision_points - current_trajectory_point
        t_coords = np.dot(centered_collisions, t)
        n_coords = np.dot(centered_collisions, n)

        # Get current tool angle in (t,n) plane
        current_angle = np.arctan2(np.dot(tool_vec, n), np.dot(tool_vec, t))

        # Determine rotation direction based on collision points position relative to tool
        cross_products = t_coords * np.dot(tool_vec, n) - n_coords * np.dot(tool_vec, t)
        if np.mean(cross_products) > 0:
            # Collisions are on the left side, rotate clockwise
            angle_increment = -np.deg2rad(1)  # About 0.6 degrees
        else:
            # Collisions are on the right side, rotate counter-clockwise
            angle_increment = np.deg2rad(1)

        test_angle = current_angle
        max_iterations = 360

        while max_iterations > 0:
            test_angle += angle_increment

            # Create new 3D tool vector by rotating in the t-n plane
            test_tool = t * np.cos(test_angle) + n * np.sin(test_angle)

            # Check against ALL candidate points to avoid creating new collisions
            cylinder_start = self.tool.get_cylinder_start(
                current_trajectory_point,
                test_tool
            )

            distances = self.compute_point_cylinder_distances(
                all_candidate_points,
                cylinder_start,
                test_tool
            )

            # If all distances are greater than tool radius, we found a valid angle
            if np.all(distances >= self.tool.radius):
                return test_angle

            max_iterations -= 1

        # No valid angle found within max iterations
        return None

    # ----------------------------------------------------------------------------------
    # 2D projection tilt angle algorithm - not converging for 40% of problematic points
    # ----------------------------------------------------------------------------------

    #def process_trajectory_for_2D_projection_algorithm(self):
    #     """Main function"""
    #     start_time = time.time()
    #     print("Trajectory processing ...")
    #
    #     # Vector acquisition
    #     t_vec, n_vec, b_vec, tool_vec = self.trajectory.calculate_local_basis()
    #
    #     print("\nCollision candidates generation ...")
    #     # Generate collision candidates for each layer
    #     for layer_idx in range(len(self.trajectory.layer_indices)):
    #         layer_points = self.trajectory.get_layer_points(layer_idx)
    #         self.collision_candidates_generator.generate_collision_candidates_per_layer(
    #             points=layer_points,
    #             normal_vectors=n_vec[layer_idx],
    #             build_vectors=b_vec[layer_idx],
    #             layer_index=layer_idx,
    #             is_last_layer=(layer_idx == len(self.trajectory.layer_indices) - 1)
    #         )
    #
    #     generation_time = time.time() - start_time
    #     print(f"\nGeneration execution time : {generation_time}")
    #
    #     # Initial collision detection
    #     print("\nInitial collision detection ...")
    #     trajectory_points_colliding = self.check_collisions(tool_vec)
    #
    #     # Get problematic points BEFORE starting the resolution
    #     problematic_points = np.where(trajectory_points_colliding)[0]
    #     print(f"\nNumber of initial trajectory points colliding: {len(problematic_points)}")
    #
    #     if len(problematic_points) == 0:
    #         print("No collisions to resolve")
    #         return tool_vec
    #
    #     # Iterative collision resolution
    #     print("\nStarting iterative collision resolution...")
    #     max_iterations = 1000
    #     updated_tool_vec = tool_vec.copy()
    #
    #     for point_idx in problematic_points:
    #         #print(f"\nProcessing point {point_idx}")
    #         point_resolved = False
    #         iteration_count = 0
    #
    #         # Vérifier que le point existe dans collision_points_set
    #         if point_idx not in self.collision_points_dict:
    #             #print(f"Warning: No collision data found for point {point_idx}")
    #             continue
    #
    #         initial_angle = np.arctan2(tool_vec[point_idx][1], tool_vec[point_idx][0])
    #         #print(f"Initial angle: {np.degrees(initial_angle):.2f}°")
    #
    #         while not point_resolved and iteration_count < max_iterations:
    #             #print(f"\nIteration {iteration_count + 1}:")
    #
    #             # Calculate new orientation based on colliding points
    #             new_angle = self.calculate_tilt_angle(
    #                 point_idx,
    #                 self.collision_points_dict[point_idx],
    #                 t_vec[point_idx],
    #                 n_vec[point_idx],
    #                 updated_tool_vec[point_idx]
    #             )
    #
    #             if new_angle is not None:
    #                 #print(f"New angle calculated: {np.degrees(new_angle):.2f}°")
    #                 #print(f"Angle change: {np.degrees(new_angle - initial_angle):.2f}°")
    #
    #                 # Update tool orientation using the new angle
    #                 updated_tool_vec[point_idx] = (
    #                         t_vec[point_idx] * np.cos(new_angle) +
    #                         n_vec[point_idx] * np.sin(new_angle)
    #                 )
    #
    #                 # Check collisions with new orientation
    #                 cylinder_start = self.tool.get_cylinder_start(
    #                     self.trajectory.points[point_idx],
    #                     updated_tool_vec[point_idx]
    #                 )
    #
    #                 distances = self.compute_point_cylinder_distances(
    #                     self.collision_candidates_dict[point_idx],
    #                     cylinder_start,
    #                     updated_tool_vec[point_idx]
    #                 )
    #
    #                 n_collisions = np.sum(distances < self.tool.radius)
    #                 #print(f"Number of remaining collisions: {n_collisions}")
    #
    #                 if not np.any(distances < self.tool.radius):
    #                     point_resolved = True
    #                     print(f"Point {point_idx} resolved with final angle {np.degrees(new_angle):.2f}°")
    #                 else:
    #                     colliding_indices = np.where(distances < self.tool.radius)[0]
    #                     self.collision_points_dict[point_idx] = self.collision_candidates_dict[point_idx][
    #                         colliding_indices]
    #             else:
    #                 #print("No valid angle found in this iteration")
    #                 pass
    #
    #             iteration_count += 1
    #
    #         if not point_resolved:
    #             print(f"Warning: Could not resolve collisions for point {point_idx} after {max_iterations} iterations")
    #
    #     return updated_tool_vec
    #
    # def calculate_tilt_angle_by_2D_projection(self, point_idx, collision_points,
    #                          t, n, tool_vec):
    #     current_trajectory_point = self.trajectory.points[point_idx]
    #
    #     # Projection on (t, n) plan
    #     centered_points = collision_points - current_trajectory_point
    #     t_coords = np.dot(centered_points, t)
    #     n_coords = np.dot(centered_points, n)
    #     projected_points = np.column_stack((t_coords, n_coords))
    #
    #     # Calculating optimal angle
    #     R = self.tool.radius
    #     angles = []
    #     for point in projected_points:
    #         d = np.linalg.norm(point)
    #         if R < d < 2 * R:
    #             point_angle = np.arctan2(point[1], point[0])
    #             delta_angle = np.arccos(R / d)
    #             angles.extend([point_angle + delta_angle, point_angle - delta_angle])
    #
    #     if not angles:
    #         return 0.0
    #
    #     best_angle = None
    #     min_valid_angle = np.inf
    #     for angle in angles:
    #         center = R * np.array([np.cos(angle), np.sin(angle)])
    #         min_distance = np.min(np.linalg.norm(projected_points - center, axis=1))
    #         if min_distance > R:
    #             angle_difference = abs(angle - np.arctan2(tool_vec[1], tool_vec[0]))
    #             angle_difference = min(angle_difference, 2 * np.pi - angle_difference)
    #             if angle_difference < min_valid_angle:
    #                 min_valid_angle = angle_difference
    #                 best_angle = angle
    #
    #     return best_angle


