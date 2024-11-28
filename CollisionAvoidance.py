import pandas as pd
import numpy as np
import trimesh
import time

# -------------------------------------------------------------------
# Trajectory acquisition class
# -------------------------------------------------------------------

class TrajectoryAcquisition:
    def __init__(self, file_path):
        # CSV reading
        self.trajectory_data = pd.read_csv(file_path)

        # Vectors extraction
        self.points = self.trajectory_data[['X', 'Y', 'Z']].values
        self.build_directions = self.trajectory_data[['Bx', 'By', 'Bz']].values
        self.tool_directions = self.trajectory_data[['Tx', 'Ty', 'Tz']].values
        self.extrusion = self.trajectory_data['Extrusion'].values
        # Layer identification
        self.layer_indices = self.identify_layers()

    def identify_layers(self):
        """Identifying starting index of each layer"""
        # Using extrusion column to identify layers
        extrusion_col = self.trajectory_data['Extrusion'].values
        layer_starts_indices = np.where(extrusion_col == 0)[0] + 1
        return layer_starts_indices

    def get_layer_points(self, layer_index):
        """Returning points of a specific layer"""
        start_idx = self.layer_indices[layer_index]
        end_idx = self.layer_indices[layer_index + 1] if layer_index + 1 < len(self.layer_indices) else len(self.points)
        return self.points[start_idx:end_idx]

    def get_layer_build_directions(self, layer_index):
        """Returning build directions of a specific layer"""
        start_idx = self.layer_indices[layer_index]
        end_idx = self.layer_indices[layer_index + 1] if layer_index + 1 < len(self.layer_indices) else len(self.points)
        return self.build_directions[start_idx:end_idx]

    def calculate_local_basis(self):
        """Vectorized local basis calculus"""
        # Tangents
        tangents = np.zeros_like(self.points)
        tangents[:-1] = np.diff(self.points, axis=0)
        tangents[-1] = tangents[-2]

        # Normalization
        b_norms = np.linalg.norm(self.build_directions, axis=1, keepdims=True)
        tool_norms = np.linalg.norm(self.tool_directions, axis=1, keepdims=True)
        t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)

        normalized_b = self.build_directions / b_norms
        normalized_t = tangents / t_norms
        normalized_tool_direction = self.tool_directions / tool_norms

        # Final vector product for n
        normalized_n = np.cross(normalized_t, normalized_b)

        return normalized_t, normalized_n, normalized_b, normalized_tool_direction

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

    def initialize_parameters(self, w_layer: float, h_layer: float):
        """Bead parameters initialization"""
        self.w_layer = w_layer
        self.h_layer = h_layer

    def generate_collision_candidates_per_layer(self, points: np.ndarray, normal_vectors: np.ndarray,build_vectors: np.ndarray,layer_index: int,is_last_layer: bool = False):
        """ Collision candidates generation """
        # Left and right points
        left_points = points + (self.w_layer / 2) * normal_vectors
        right_points = points - (self.w_layer / 2) * normal_vectors

        # Collision candidates structure update
        if self.left_points is None:
            self.left_points = left_points
            self.right_points = right_points
        else:
            self.left_points = np.vstack((self.left_points, left_points))
            self.right_points = np.vstack((self.right_points, right_points))

        # Last layer case
        if is_last_layer:
            top_points = points + self.h_layer * build_vectors
            self.top_points = top_points

        # To indicate the layer corresponding to the collision candidates
        self.layer_indices.append(len(self.left_points) - len(points))
        self.current_layer = layer_index

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
    def __init__(self, trajectory_data, bead_width, bead_height, tool_radius, tool_length):
        # Instances creation
        self.trajectory = trajectory_data

        self.collision_candidates_generator = CollisionCandidatesGenerator()
        self.collision_candidates_generator.initialize_parameters(bead_width, bead_height)

        self.tool = Tool(radius=tool_radius, nozzle_length=tool_length)

    def process_trajectory(self):
        """Main function"""
        start_time = time.time() # for optimization purposes
        print("Trajectory processing ...")

        # Vector acquisition
        t_vec, n_vec, b_vec, tool_vec = self.trajectory.calculate_local_basis()

        print("\nCollision candidates generation ...")
        # For each layer
        for layer_idx in range(len(self.trajectory.layer_indices)):
            # Layer points acquisition
            layer_points = self.trajectory.get_layer_points(layer_idx)
            # Collision candidates generation
            self.collision_candidates_generator.generate_collision_candidates_per_layer(
                points=layer_points,
                normal_vectors=n_vec[layer_idx],
                build_vectors=b_vec[layer_idx],
                layer_index=layer_idx,
                is_last_layer=(layer_idx == len(self.trajectory.layer_indices) - 1)
            )
        generation_time = time.time() - start_time
        print(f"\nGeneration execution time : {generation_time}")

        print("\nCollision detection ...")
        trajectory_points_colliding = self.check_collisions(tool_vec)
        checking_time = time.time() - start_time
        print(f"\nCollision detection execution time : {checking_time}")

        trajectory_points_colliding_index = np.where(trajectory_points_colliding)[0]
        print(f"\nNumber of trajectory points colliding: {len(trajectory_points_colliding_index)}")

        pass


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
        self.collision_points_set = {}

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
                        self.collision_points_set [point_idx] = points_to_check.copy()
        return collisions_detected

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

    def calculate_tilt_angle(self, point_idx, collision_points,
                             t, n, tool_vec):
        current_trajectory_point = self.trajectory.points[point_idx]

        # Projection on (t, n) plan
        centered_points = collision_points - current_trajectory_point
        t_coords = np.dot(centered_points, t)
        n_coords = np.dot(centered_points, n)
        projected_points = np.column_stack((t_coords, n_coords))

        # Calculating optimal angle
        R = self.tool.radius
        angles = []
        for point in projected_points:
            d = np.linalg.norm(point)
            if R < d < 2 * R:
                point_angle = np.arctan2(point[1], point[0])
                delta_angle = np.arccos(R / d)
                angles.extend([point_angle + delta_angle, point_angle - delta_angle])

        if not angles:
            return 0.0

        best_angle = None
        min_valid_angle = np.inf
        for angle in angles:
            center = R * np.array([np.cos(angle), np.sin(angle)])
            min_distance = np.min(np.linalg.norm(projected_points - center, axis=1))
            if min_distance > R:
                angle_difference = abs(angle - np.arctan2(tool_vec[1], tool_vec[0]))
                angle_difference = min(angle_difference, 2 * np.pi - angle_difference)
                if angle_difference < min_valid_angle:
                    min_valid_angle = angle_difference
                    best_angle = angle

        return best_angle