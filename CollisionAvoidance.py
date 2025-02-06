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

    def generate_collision_candidates_per_layer(self, points, normal_vectors, build_vectors, layer_index,
                                                is_last_layer):
        """Generating collision candidates for a layer"""
        # Checking if the layer was already treated
        if layer_index in self.already_generated:
            return

        # Calculating left and right points
        left_points = points + (self.w_layer / 2) * normal_vectors
        right_points = points - (self.w_layer / 2) * normal_vectors

        # Calculating top points for current layer
        top_points = points + self.h_layer * build_vectors

        # Update collision candidates
        if self.left_points is None:
            self.left_points = left_points
            self.right_points = right_points
            self.top_points = top_points
        else:
            self.left_points = np.vstack((self.left_points, left_points))
            self.right_points = np.vstack((self.right_points, right_points))
            self.top_points = np.vstack((self.top_points, top_points))

        # Updating layer indices
        self.layer_indices.append(len(self.left_points) - len(points))
        self.current_layer = layer_index

        # Mark the layer as treated
        self.already_generated[layer_index] = True

    def get_all_current_collision_candidates(self):
        """Return current collision candidates"""
        return np.vstack((self.left_points, self.right_points, self.top_points))

    def get_layer_collision_candidates(self, layer_start_idx, layer_end_idx):
        """Return collision candidates for a specific layer range"""
        left = self.left_points[layer_start_idx:layer_end_idx]
        right = self.right_points[layer_start_idx:layer_end_idx]
        top = self.top_points[layer_start_idx:layer_end_idx]
        return left, right, top

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
    def __init__(self, trajectory_path, bead_width, bead_height, tool_radius, tool_length, nb_previous_layers=5):
        # Create trajectory manager instance
        self.trajectory = TrajectoryManager(trajectory_path)
        self.trajectory.compute_and_store_local_bases()

        # Collision candidates generator
        self.collision_candidates_generator = CollisionCandidatesGenerator()
        self.collision_candidates_generator.initialize_parameters(bead_width, bead_height)

        self.tool = Tool(radius=tool_radius, nozzle_length=tool_length)

        # Pour stocker les informations de collision
        self.collision_points_dict = {}
        self.all_candidates_dict = {}  # Nouveau dictionnaire
        self._collision_points = None

    def get_collision_indices(self):
        """Retourne les indices des points de la trajectoire qui sont en collision"""
        if self._collision_points is None:
            # Si la détection n'a pas encore été faite, on la fait
            self.detect_collisions_optimized()

        # Retourne les indices où collision_points est True
        return np.where(self._collision_points)[0]

    def detect_collisions_optimized(self):
        """Version optimisée de la détection de collisions"""
        if self._collision_points is not None:
            return self._collision_points

        # Pré-calculs
        avg_build_vectors = self.trajectory.compute_average_build_vectors()
        avg_build_vectors /= np.linalg.norm(avg_build_vectors, axis=1)[:, np.newaxis]

        # Pré-calcul des ranges de couches
        layer_ranges = [(self.trajectory.layer_indices[i],
                         self.trajectory.layer_indices[i + 1] if i + 1 < len(self.trajectory.layer_indices)
                         else len(self.trajectory.points))
                        for i in range(len(self.trajectory.layer_indices))]

        # Génération initiale des points candidats
        for layer_idx in range(len(self.trajectory.layer_indices)):
            layer_points = self.trajectory.get_layer_points(layer_idx)
            self.collision_candidates_generator.generate_collision_candidates_per_layer(
                points=layer_points,
                normal_vectors=self.trajectory.n_vectors[layer_idx],
                build_vectors=self.trajectory.b_vectors[layer_idx],
                layer_index=layer_idx,
                is_last_layer=(layer_idx == len(self.trajectory.layer_indices) - 1)
            )

        collision_points = np.zeros(len(self.trajectory.points), dtype=bool)
        self.collision_points_dict = {}
        self.all_candidates_dict = {}

        # Allocation des tableaux temporaires
        max_points = max(len(range(*r)) for r in layer_ranges)
        diff_vectors = np.empty((max_points * 2, 3))  # *2 pour left et right points

        for point_idx, current_point in enumerate(self.trajectory.points):
            current_layer = np.searchsorted(self.trajectory.layer_indices[1:], point_idx)

            # Points en collision pour ce point
            collision_mask = None
            all_points = None

            for layer in range(current_layer + 1):
                start_idx, end_idx = layer_ranges[layer]
                if layer == current_layer:
                    end_idx = min(end_idx, point_idx)

                # Récupération et concaténation des points
                left, right, top = self.collision_candidates_generator.get_layer_collision_candidates(
                    start_idx, end_idx)

                if len(left) > 0 or len(right) > 0:
                    # Concaténation efficace
                    test_points = np.vstack([p for p in [left, right] if len(p) > 0])

                    # Calcul vectorisé des projections
                    np.subtract(test_points, current_point, out=diff_vectors[:len(test_points)])
                    projections = np.abs(np.dot(diff_vectors[:len(test_points)],
                                                avg_build_vectors[layer]))

                    # Mise à jour du masque de collision
                    point_mask = projections <= self.tool.radius * 1.1

                    if np.any(point_mask):
                        if collision_mask is None:
                            collision_mask = point_mask
                            all_points = test_points
                        else:
                            collision_mask = np.concatenate([collision_mask, point_mask])
                            all_points = np.vstack([all_points, test_points])

                # Ajout direct des top points pour la couche courante
                if layer == current_layer and len(top) > 0:
                    if collision_mask is None:
                        collision_mask = np.ones(len(top), dtype=bool)
                        all_points = top
                    else:
                        collision_mask = np.concatenate([collision_mask, np.ones(len(top), dtype=bool)])
                        all_points = np.vstack([all_points, top])

            # Vérification finale des collisions
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
        """Détection exhaustive sans filtrage."""
        if self._collision_points is not None:
            return self._collision_points

        collision_points = np.zeros(len(self.trajectory.points), dtype=bool)
        self.collision_points_dict = {}
        self.all_candidates_dict = {}

        # Génération des points candidats pour toutes les couches
        for layer_idx in range(len(self.trajectory.layer_indices)):
            layer_points = self.trajectory.get_layer_points(layer_idx)
            self.collision_candidates_generator.generate_collision_candidates_per_layer(
                points=layer_points,
                normal_vectors=self.trajectory.n_vectors[layer_idx],
                build_vectors=self.trajectory.b_vectors[layer_idx],
                layer_index=layer_idx,
                is_last_layer=(layer_idx == len(self.trajectory.layer_indices) - 1)
            )

        for point_idx, current_point in enumerate(self.trajectory.points):
            current_layer = np.searchsorted(self.trajectory.layer_indices[1:], point_idx)
            collision_candidates = []

            # Vérification de chaque couche jusqu'à la courante
            for layer in range(current_layer + 1):
                start_idx = self.trajectory.layer_indices[layer]
                end_idx = (self.trajectory.layer_indices[layer + 1]
                           if layer + 1 < len(self.trajectory.layer_indices)
                           else len(self.trajectory.points))

                if layer == current_layer:
                    end_idx = min(end_idx, point_idx)

                left_points, right_points, top_points = \
                    self.collision_candidates_generator.get_layer_collision_candidates(start_idx, end_idx)

                if len(left_points) > 0:
                    collision_candidates.append(left_points)
                if len(right_points) > 0:
                    collision_candidates.append(right_points)

                if layer == current_layer and len(top_points) > 0:
                    collision_candidates.append(top_points)

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
        trajectory_points_colliding = self.detect_collisions_optimized()
        #trajectory_points_colliding = self.detect_collisions_exhaustive()

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
                    self.all_candidates_dict[point_idx],
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

    def verify_collisions(self, updated_tool_vectors=None):
        """Vérifie les collisions résiduelles après modification des vecteurs outils"""
        tool_vectors = updated_tool_vectors if updated_tool_vectors is not None else self.trajectory.tool_vectors
        collision_count = 0
        residual_collisions = []

        # Vérifier uniquement les points précédemment en collision
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
    # Algorithm for tool_angle modification using auxiliary functions
    # ----------------------------------------------------------------------------------

    def calculate_tilt_angle(self, point_idx, collision_points, t, n, b, tool_vec):
        """
        Calcule l'angle de tilt optimal en deux phases:
        1. Recherche grossière pour trouver une première solution
        2. Optimisation fine en revenant vers l'angle initial
        """
        current_point = self.trajectory.points[point_idx]
        all_candidate_points = self.all_candidates_dict[point_idx]

        # Point de départ : vecteur outil actuel
        current_angle = np.arctan2(np.dot(tool_vec, b), np.dot(tool_vec, n))

        # Analyse du quadrant prioritaire
        centered_points = collision_points - current_point
        projected_points = np.column_stack((
            np.dot(centered_points, n),
            np.dot(centered_points, b)
        ))

        # Comptage des points par quadrant
        angle_diffs = np.arctan2(projected_points[:, 1], projected_points[:, 0]) - current_angle
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi  # Normalisation [-pi, pi]
        quad1_count = np.sum((0 < angle_diffs) & (angle_diffs < np.pi / 2))
        quad2_count = np.sum((-np.pi / 2 < angle_diffs) & (angle_diffs < 0))

        direction = 1 if quad2_count > quad1_count else -1

        # Phase 1: Recherche grossière (recherche dichotomique)
        coarse_angles = np.radians([1, 15, 30, 45, 60, 90])
        first_solution = None

        for angle in coarse_angles:
            test_angle = current_angle + direction * angle
            test_tool_vec = n * np.cos(test_angle) + b * np.sin(test_angle)

            distances = self.compute_point_cylinder_distances(
                all_candidate_points,
                self.tool.get_cylinder_start(current_point, test_tool_vec),
                test_tool_vec
            )

            if np.all(distances >= self.tool.radius):
                first_solution = test_angle
                break

        if first_solution is None:
            return None

        # Phase 2: Optimisation fine (adaptative)
        fine_step = np.radians(1.0)  # Pas initial plus large
        current_test_angle = first_solution
        best_valid_angle = first_solution

        while fine_step > np.radians(0.1):  # Réduction progressive du pas
            test_angle = current_test_angle - direction * fine_step
            test_tool_vec = n * np.cos(test_angle) + b * np.sin(test_angle)

            distances = self.compute_point_cylinder_distances(
                all_candidate_points,
                self.tool.get_cylinder_start(current_point, test_tool_vec),
                test_tool_vec
            )

            if np.all(distances >= self.tool.radius):
                best_valid_angle = test_angle
                current_test_angle = test_angle
            else:
                fine_step /= 2  # Réduire le pas si une collision est trouvée

        return best_valid_angle