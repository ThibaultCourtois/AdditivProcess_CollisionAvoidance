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

        # Initialisation des statistiques
        unresolved_points_data = {}
        start_transition_points = 0
        resolved_transition_points = 0

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
        updated_tool_vec = tool_vec.copy()
        total_resolved = 0

        initial_angle_range = np.pi / 6  # 30 degrés
        max_angle_range = np.pi  # 90 degrés
        angle_increment = np.pi / 12  # 15 degrés

        for point_idx in problematic_points:
            if point_idx not in self.collision_points_dict:
                continue

            # Get all candidates for this point
            all_candidates = self.collision_candidates_dict[point_idx]
            collision_points = self.collision_points_dict[point_idx]

            # Initial angle calculation
            current_angle = np.arctan2(np.dot(tool_vec[point_idx], n_vec[point_idx]),
                                       np.dot(tool_vec[point_idx], t_vec[point_idx]))

            # Point analysis
            analysis = self.analyze_problematic_point(
                point_idx, collision_points,
                t_vec[point_idx], n_vec[point_idx],
                current_angle
            )

            # Transition points check
            is_transition = self.is_transition_point(collision_points, t_vec[point_idx], n_vec[point_idx])
            if is_transition:
                start_transition_points += 1

            # Increasing angle range
            angle_range = initial_angle_range
            new_angle = None

            while new_angle is None and angle_range <= max_angle_range:
                new_angle = self.calculate_tilt_angle(
                    point_idx,
                    collision_points,
                    all_candidates,
                    t_vec[point_idx],
                    n_vec[point_idx],
                    updated_tool_vec[point_idx],
                    angle_range
                )

                if new_angle is None:
                    angle_range += angle_increment

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

                angle_sector = new_angle - current_angle
                while angle_sector > np.pi:
                    angle_sector -= 2 * np.pi
                while angle_sector < -np.pi:
                    angle_sector += 2 * np.pi

                if not np.any(distances < self.tool.radius):
                    total_resolved += 1
                    if is_transition:
                        resolved_transition_points += 1
                else:
                    unresolved_points_data[point_idx] = {
                        'analysis': analysis,
                        'is_transition': is_transition,
                        'num_collisions': np.sum(distances < self.tool.radius)
                    }
            else:
                print(f"Warning: No valid angle found for point {point_idx}")
                unresolved_points_data[point_idx] = {
                    'analysis': analysis,
                    'is_transition': is_transition,
                    'reason': 'No valid angle found'
                }

        # Final statistics
        resolution_time = time.time() - start_time
        print(f"\nTotal points resolved: {total_resolved}/{len(problematic_points)}")
        print(f"Total execution time: {resolution_time:.2f} seconds")

        print(f"\nTransition Points Statistics:")
        print(f"Initial transition points: {start_transition_points}")
        print(f"Resolved transition points: {resolved_transition_points}")

        print(f"\nUnresolved Points Analysis:")
        print(f"Total unresolved points: {len(unresolved_points_data)}")

        transition_unresolved = sum(1 for data in unresolved_points_data.values()
                                    if data['is_transition'])
        print(f"Unresolved transition points: {transition_unresolved}")

        # Unresolved points analysis
        if unresolved_points_data:
            print("\nDetails of unresolved points:")
            for point_idx, data in unresolved_points_data.items():
                print(f"\nPoint {point_idx}:")
                print(f"Is transition point: {data['is_transition']}")
                if 'num_collisions' in data:
                    print(f"Remaining collisions: {data['num_collisions']}")
                print(f"Angle range: [{np.min(data['analysis']['angles']):.1f}°, "
                      f"{np.max(data['analysis']['angles']):.1f}°]")
                print(f"Point density: {data['analysis']['density']:.2f} points/degree")

        return updated_tool_vec

    # ----------------------------------------------------------------------------------
    # Algorithm for tool_angle modification using auxiliary functions
    # ----------------------------------------------------------------------------------

    def calculate_tilt_angle(self, point_idx, collision_points, all_candidate_points, t, n, tool_vec, angle_range):
        """
        Calcule l'angle optimal pour orienter l'outil en évitant les collisions.
        """
        current_point = self.trajectory.points[point_idx]
        current_angle = np.arctan2(np.dot(tool_vec, n), np.dot(tool_vec, t))

        # Type of problematic point
        is_transition = self.is_transition_point(collision_points, t, n)

        # 1. Angle range for testing calcultation
        limiting_angles = self.calculate_limiting_angles(collision_points, current_point, t, n, angle_range)
        if limiting_angles is None or len(limiting_angles) == 0:
            return None

        # 2. Cost function definition
        def cost_function(angle):
            test_tool = t * np.cos(angle) + n * np.sin(angle)

            # Collision cost
            distances = self.compute_point_cylinder_distances(
                all_candidate_points,
                self.tool.get_cylinder_start(current_point, test_tool),
                test_tool
            )
            collision_cost = np.sum(np.maximum(0, self.tool.radius - distances))

            # Weight adjustment depending on problematic point
            if is_transition:
                return (20.0 * collision_cost +
                        0.5 * abs(angle - current_angle) +
                        8.0 * self.compute_continuity_cost(point_idx, angle))
            else:
                return (10.0 * collision_cost +
                        2.0 * abs(angle - current_angle) +
                        1.0 * self.compute_continuity_cost(point_idx, angle))

        # 3. Binary search of optimal tilt angle using cost unction and test angle range
        best_angle = self.binary_search_angles(limiting_angles, cost_function)

        # 4. Final verification
        if best_angle is not None:
            test_tool = t * np.cos(best_angle) + n * np.sin(best_angle)
            final_distances = self.compute_point_cylinder_distances(
                all_candidate_points,
                self.tool.get_cylinder_start(current_point, test_tool),
                test_tool
            )

            if np.all(final_distances >= self.tool.radius):
                return best_angle

        return None

    def calculate_limiting_angles(self, collision_points, center, t, n, angle_range=np.pi / 6):
        if len(collision_points) < 10:
            angles = np.array([np.arctan2(np.dot(vec, n), np.dot(vec, t))
                               for vec in collision_points])
            mean_angle = np.mean(angles)

            # Test des deux côtés
            opposite_angles = [mean_angle + np.pi - angle_range, mean_angle + np.pi + angle_range]
            same_side_angles = [mean_angle - angle_range, mean_angle + angle_range]

            # Évaluer le meilleur côté
            best_score = -1
            best_angles = opposite_angles

            for test_angles in [opposite_angles, same_side_angles]:
                test_tool = t * np.cos(np.mean(test_angles)) + n * np.sin(np.mean(test_angles))
                distances = self.compute_point_cylinder_distances(
                    collision_points,
                    self.tool.get_cylinder_start(center, test_tool),
                    test_tool
                )
                score = np.min(distances) / self.tool.radius
                if score > best_score:
                    best_score = score
                    best_angles = test_angles

            return np.array(best_angles)

        # For standard points, we analyse the collision_points repartition and we chose the opposite direction
        vectors_to_collisions = collision_points - center
        mean_vector = np.mean(vectors_to_collisions, axis=0)
        mean_direction = mean_vector / np.linalg.norm(mean_vector)

        t_proj = np.dot(mean_direction, t)
        n_proj = np.dot(mean_direction, n)
        base_angle = np.arctan2(n_proj, t_proj) + np.pi

        return np.array([base_angle - angle_range, base_angle + angle_range])

    def calculate_transition_angles(self, collision_points, center, t, n):
        """
        Calcule les secteurs angulaires sans collision pour les points de transition
        """
        vectors_to_collisions = collision_points - center
        angles = np.array([np.arctan2(np.dot(vec, n), np.dot(vec, t))
                           for vec in vectors_to_collisions])

        sorted_angles = np.sort(angles)

        # Biggest gap without collision
        gaps = np.diff(np.append(sorted_angles, sorted_angles[0] + 2 * np.pi))
        max_gap_idx = np.argmax(gaps)

        # Middle of the biggest gap
        optimal_angle = sorted_angles[max_gap_idx] + gaps[max_gap_idx] / 2
        if optimal_angle > 2 * np.pi:
            optimal_angle -= 2 * np.pi

        sector_size = min(gaps[max_gap_idx] / 3, np.pi / 4)  # On prend le tiers du gap ou 45° max
        return [optimal_angle - sector_size, optimal_angle + sector_size]

    def compute_continuity_cost(self, point_idx, angle):
        """
        Calculate continuity cost
        """
        cost = 0
        window_size = 3

        # Checking previous points
        for i in range(1, window_size + 1):
            if point_idx - i >= 0:
                prev_tool = self.trajectory.tool_directions[point_idx - i]
                prev_angle = np.arctan2(
                    np.dot(prev_tool, self.trajectory.build_directions[point_idx - i]),
                    np.dot(prev_tool, self.trajectory.tool_directions[point_idx - i])
                )
                cost += (1.0 / i) * abs(angle - prev_angle)

        # Checking next points
        for i in range(1, window_size + 1):
            if point_idx + i < len(self.trajectory.points):
                next_tool = self.trajectory.tool_directions[point_idx + i]
                next_angle = np.arctan2(
                    np.dot(next_tool, self.trajectory.build_directions[point_idx + i]),
                    np.dot(next_tool, self.trajectory.tool_directions[point_idx + i])
                )
                cost += (1.0 / i) * abs(angle - next_angle)

        return cost

    def binary_search_angles(self, angles, cost_function, n_intervals=10):
        """
        Optimized research of the optimal angle
        """
        if angles is None:
            return None

        best_angle = None
        min_cost = float('inf')

        # Normalisation des angles entre 0 et 2π
        angles = np.sort(angles) % (2 * np.pi)

        # Regroupement des angles proches (optimisation)
        grouped_angles = []
        current_group = [angles[0]]

        for angle in angles[1:]:
            if abs(angle - current_group[-1]) < np.deg2rad(5):
                current_group.append(angle)
            else:
                grouped_angles.append(np.mean(current_group))
                current_group = [angle]
        if current_group:
            grouped_angles.append(np.mean(current_group))

        # Création des intervalles optimisés
        intervals = []
        for i in range(0, len(grouped_angles), 2):
            if i + 1 < len(grouped_angles):
                start, end = grouped_angles[i], grouped_angles[i + 1]
                if start > end:
                    # Gestion du passage par 0
                    if end > np.pi:
                        intervals.append((start, end))
                    else:
                        intervals.append((start, 2 * np.pi))
                        intervals.append((0, end))
                else:
                    intervals.append((start, end))

        # Recherche optimisée dans les intervalles
        for start, end in intervals:
            # Première passe avec peu de points
            coarse_angles = np.linspace(start, end, 5)
            coarse_costs = [cost_function(angle) for angle in coarse_angles]
            best_coarse_idx = np.argmin(coarse_costs)

            # Raffinement autour du meilleur angle grossier
            if best_coarse_idx > 0 and best_coarse_idx < len(coarse_angles) - 1:
                fine_start = coarse_angles[best_coarse_idx - 1]
                fine_end = coarse_angles[best_coarse_idx + 1]
            else:
                fine_start = coarse_angles[best_coarse_idx]
                fine_end = coarse_angles[best_coarse_idx]

            fine_angles = np.linspace(fine_start, fine_end, n_intervals)
            for angle in fine_angles:
                cost = cost_function(angle)
                if cost < min_cost:
                    min_cost = cost
                    best_angle = angle

        return best_angle

    def is_transition_point(self, collision_points, t, n):
        if len(collision_points) < 3:
            return False

        angles = np.array([np.arctan2(np.dot(vec, n), np.dot(vec, t))
                           for vec in collision_points])
        sorted_angles = np.sort(angles) % (2 * np.pi)

        max_gap = np.max(np.diff(sorted_angles))
        angle_range = np.max(angles) - np.min(angles)
        point_density = len(angles) / (angle_range + 1e-6)

        return (max_gap > np.pi / 2 or
                (angle_range > np.pi and point_density < 0.5) or
                len(collision_points) > 100)

    def analyze_problematic_point(self, point_idx, collision_points, t, n, current_angle):
        """
        Debug analysis
        """
        print(f"\n=== Detailed Analysis for Point {point_idx} ===")

        # Distribution spatiale
        vectors_to_collisions = collision_points - self.trajectory.points[point_idx]
        distances = np.linalg.norm(vectors_to_collisions, axis=1)

        print(f"Spatial Distribution:")
        print(f"Min distance: {np.min(distances):.2f}")
        print(f"Max distance: {np.max(distances):.2f}")
        print(f"Mean distance: {np.mean(distances):.2f}")

        # Distribution angulaire
        angles = np.array([np.arctan2(np.dot(vec, n), np.dot(vec, t))
                           for vec in vectors_to_collisions])
        angles = np.degrees(angles)

        print(f"\nAngular Distribution:")
        print(f"Angle range: [{np.min(angles):.1f}°, {np.max(angles):.1f}°]")
        print(f"Current tool angle: {np.degrees(current_angle):.1f}°")

        # Densité des points de collision
        angle_density = len(angles) / (np.max(angles) - np.min(angles))
        print(f"Point density: {angle_density:.2f} points/degree")

        # Ajout d'analyse de configuration
        vectors_to_collisions = collision_points - self.trajectory.points[point_idx]
        radial_distances = np.dot(vectors_to_collisions, t)
        axial_distances = np.dot(vectors_to_collisions, n)

        print("\nConfiguration Analysis:")
        print(f"Radial extent: {np.min(radial_distances):.2f} to {np.max(radial_distances):.2f}")
        print(f"Axial extent: {np.min(axial_distances):.2f} to {np.max(axial_distances):.2f}")
        print(
            f"Radial/Axial ratio: {(np.max(radial_distances) - np.min(radial_distances)) / (np.max(axial_distances) - np.min(axial_distances)):.2f}")

        return {
            'distances': distances,
            'angles': angles,
            'density': angle_density,
            'radial_extent': (np.min(radial_distances), np.max(radial_distances)),
            'axial_extent': (np.min(axial_distances), np.max(axial_distances))
        }


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


