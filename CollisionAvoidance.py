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
        self.trajectory.compute_and_store_local_bases()  # Calcul des bases au chargement

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

        # Utiliser directement les bases pré-calculées
        for layer_idx in range(len(self.trajectory.layer_indices)):
            layer_points = self.trajectory.get_layer_points(layer_idx)
            self.collision_candidates_generator.generate_collision_candidates_per_layer(
                points=layer_points,
                normal_vectors=self.trajectory.n_vectors[layer_idx],
                build_vectors=self.trajectory.b_vectors[layer_idx],
                layer_index=layer_idx,
                is_last_layer=(layer_idx == len(self.trajectory.layer_indices) - 1)
            )

        self._collision_points = self.check_collisions(self.trajectory.tool_vectors)
        self._initial_collisions_detected = True

        return self._collision_points

    def verify_collisions(self, updated_tool_vectors=None):
        """Vérifie les collisions sur l'ensemble de la trajectoire avec les vecteurs outils actuels"""
        collision_count = 0
        residual_collisions = []

        # Si pas de vecteurs fournis, utiliser ceux de la trajectoire
        tool_vectors = updated_tool_vectors if updated_tool_vectors is not None else self.trajectory.tool_vectors

        # Check initial collisions
        initial_collision_points = np.where(self._collision_points)[0]

        for point_idx in initial_collision_points:
            if point_idx in self.collision_candidates_dict:
                # Recalculer la distance avec le vecteur outil actuel
                distances = self.compute_point_cylinder_distances(
                    self.collision_candidates_dict[point_idx],
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
        """Fonction principale de traitement de la trajectoire"""
        print("Trajectory processing...")
        start_time = time.time()

        # Utiliser directement les vecteurs de base pré-calculés
        t_vec = self.trajectory.t_vectors
        n_vec = self.trajectory.n_vectors
        b_vec = self.trajectory.b_vectors
        tool_vec = self.trajectory.tool_vectors.copy()
        updated_tool_vec = tool_vec.copy()

        # Détection des collisions initiales
        trajectory_points_colliding = self.detect_initial_collisions()

        # Points problématiques
        problematic_points = np.where(trajectory_points_colliding)[0]
        print(f"\nNumber of initial trajectory points colliding: {len(problematic_points)}")

        if len(problematic_points) == 0:
            print("No collisions to resolve")
            return tool_vec

        # Résolution des collisions
        total_resolved = 0
        correction_angles = []  # Pour calculer la moyenne
        print("\nStarting collision resolution:")
        print("-" * 50)

        # Traitement de chaque point problématique
        for point_idx in problematic_points:
            if point_idx not in self.collision_points_dict:
                print(f"Point {point_idx}: Skipped (no collision data)")
                continue

            print(f"\nProcessing point {point_idx}:")

            # Angle initial
            initial_angle = np.arctan2(
                np.dot(updated_tool_vec[point_idx], b_vec[point_idx]),
                np.dot(updated_tool_vec[point_idx], n_vec[point_idx])
            )

            # Tentative de résolution
            new_angle = self.calculate_tilt_angle(
                point_idx,
                self.collision_points_dict[point_idx],
                self.collision_candidates_dict[point_idx],
                t_vec[point_idx],
                n_vec[point_idx],
                b_vec[point_idx],
                updated_tool_vec[point_idx]
            )

            if new_angle is not None:
                # Mise à jour du vecteur outil
                updated_tool_vec[point_idx] = (
                        n_vec[point_idx] * np.cos(new_angle) +
                        b_vec[point_idx] * np.sin(new_angle)
                )

                # Vérification de la résolution
                distances = self.compute_point_cylinder_distances(
                    self.collision_candidates_dict[point_idx],
                    self.tool.get_cylinder_start(
                        self.trajectory.points[point_idx],
                        updated_tool_vec[point_idx]
                    ),
                    updated_tool_vec[point_idx]
                )

                if np.all(distances >= self.tool.radius):
                    total_resolved += 1
                    # Calcul de l'amplitude de correction
                    angle_diff = new_angle - initial_angle
                    while angle_diff > np.pi:
                        angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi:
                        angle_diff += 2 * np.pi

                    correction_angles.append(abs(angle_diff))
                    print(f"Point {point_idx}: RESOLVED")
                    print(f"   Correction amplitude: {np.degrees(abs(angle_diff)):.2f}°")
                else:
                    print(f"Point {point_idx}: FAILED (Collision still present after angle correction)")
            else:
                print(f"Point {point_idx}: FAILED (No valid angle found)")

        final_collision_count, residual_points = self.verify_collisions(updated_tool_vec)
        print("\n" + "-" * 50)
        print(f"Total points resolved: {len(problematic_points) - final_collision_count}/{len(problematic_points)}")
        print(
            f"Resolution rate: {((len(problematic_points) - final_collision_count) / len(problematic_points) * 100):.1f}%")
        if correction_angles:
            mean_correction = np.degrees(np.mean(correction_angles))
            print(f"Average correction amplitude: {mean_correction:.2f}°")
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")

        print(f"\nVerification finale:")
        print(f"Nombre de collisions résiduelles: {final_collision_count}")
        print(f"Points problématiques: {residual_points}")

        return updated_tool_vec

    # ----------------------------------------------------------------------------------
    # Algorithm for tool_angle modification using auxiliary functions
    # ----------------------------------------------------------------------------------

    def calculate_tilt_angle(self, point_idx, collision_points, all_candidate_points, t, n, b, tool_vec):
        """
        Calcule l'angle de tilt minimal pour éviter les collisions en projetant les points dans le plan (n,b)
        de manière itérative en minimisant le nombre de collisions.
        """
        current_point = self.trajectory.points[point_idx]
        max_iterations = 10

        # Angles de test non linéaires
        angles = np.concatenate([
            np.linspace(0.1, np.pi / 6, 15),  # 15 points entre 0.1 et 30°
            np.linspace(np.pi / 6, np.pi / 2, 10)  # 10 points entre 30° et 90°
        ])

        # Point de départ : vecteur outil actuel
        best_tool_vec = tool_vec
        current_angle = np.arctan2(np.dot(tool_vec, b), np.dot(tool_vec, n))
        min_collisions = len(all_candidate_points)  # Pire cas initial

        for iteration in range(max_iterations):
            if iteration == 0:
                print(f"    Initial attempt...")
            else:
                print(f"    Iteration {iteration} (with {min_collisions} collisions)...")

            # Analyse de la distribution des points dans les quadrants
            centered_points = collision_points - current_point
            n_coords = np.dot(centered_points, n)
            b_coords = np.dot(centered_points, b)
            projected_points = np.column_stack((n_coords, b_coords))

            quad1_count = 0  # Quadrant sens horaire
            quad2_count = 0  # Quadrant sens anti-horaire

            for point in projected_points:
                point_angle = np.arctan2(point[1], point[0])
                angle_diff = point_angle - current_angle

                # Normaliser entre -π et π
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi

                if 0 < angle_diff < np.pi / 2:
                    quad1_count += 1
                elif -np.pi / 2 < angle_diff < 0:
                    quad2_count += 1

            # Direction préférentielle
            preferred_direction = 1 if quad2_count > quad1_count else -1

            # Test des deux quadrants
            found_solution = False
            best_angle_this_iter = current_angle

            for direction in [preferred_direction, -preferred_direction]:
                for test_angle_offset in angles:
                    test_angle = current_angle + direction * test_angle_offset
                    test_tool_vec = n * np.cos(test_angle) + b * np.sin(test_angle)

                    # Vérifier les collisions
                    distances = self.compute_point_cylinder_distances(
                        all_candidate_points,
                        self.tool.get_cylinder_start(current_point, test_tool_vec),
                        test_tool_vec
                    )

                    num_collisions = np.sum(distances < self.tool.radius)

                    # Si solution sans collision trouvée
                    if num_collisions == 0:
                        return test_angle

                    # Sinon, garder le meilleur cas
                    if num_collisions < min_collisions:
                        min_collisions = num_collisions
                        best_angle_this_iter = test_angle
                        best_tool_vec = test_tool_vec

            # Mise à jour pour la prochaine itération
            if best_angle_this_iter == current_angle:
                # Aucune amélioration trouvée
                break

            current_angle = best_angle_this_iter

        return None





