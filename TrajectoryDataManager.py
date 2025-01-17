import pandas as pd
import numpy as np
import scipy
# -------------------------------------------------------------------
# For trajectory acquisition and modification purposes
# -------------------------------------------------------------------

class TrajectoryManager:
    def __init__(self, file_path=None):
        self.trajectory_data = None
        self.points = None
        self.build_directions = None
        self.tool_directions = None
        self.extrusion = None
        self.layer_indices = None

        # Nouvelles propriétés pour stocker les bases locales
        self.t_vectors = None
        self.n_vectors = None
        self.b_vectors = None
        self.tool_vectors = None

        if file_path:
            self.load_trajectory(file_path)
            # Calculer les bases locales au chargement
            self.compute_and_store_local_bases()

    def compute_and_store_local_bases(self):
        """Calculate and store local bases, excluding non-extrusion points"""
        # Calculate initial basis only for points with extrusion
        t_vec, n_vec, b_vec, tool_vec = self.calculate_local_basis()

        # Store results
        self.t_vectors = t_vec
        self.n_vectors = n_vec
        self.b_vectors = b_vec
        self.tool_vectors = tool_vec

        # Fix sharp turns after initial basis calculation
        self.fix_extremities_vector()
        self.fix_symmetry_transitions()

    def load_trajectory(self, file_path):
        """Load and process trajectory data from file"""
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

    def detect_extremities(self):
        """Détecte les points avec des sauts importants en distance et direction"""
        # Calculer les différences de position
        point_diffs = np.diff(self.points, axis=0)
        distances = np.linalg.norm(point_diffs, axis=1)

        # Calculer la distance moyenne et son écart-type
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # Seuil de distance plus adapté basé sur la statistique des distances
        dist_threshold = mean_dist + 3 * std_dist  # règle des 3-sigma

        # Calculer les angles entre vecteurs consécutifs quand ils existent
        directions = np.zeros_like(point_diffs)
        mask = distances > 1e-10
        directions[mask] = point_diffs[mask] / distances[mask, np.newaxis]
        dot_products = np.sum(directions[:-1] * directions[1:], axis=1)
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))

        # Créer le masque initial sur les distances
        sharp_turn_mask = np.zeros(len(self.points), dtype=bool)

        # Identifier les points avec des grandes distances
        large_dist_indices = np.where(distances > dist_threshold)[0]

        # Pour chaque point avec une grande distance, vérifier si c'est aussi
        # un point avec un changement significatif de direction
        for idx in large_dist_indices:
            if idx > 0 and idx < len(angles):  # Vérifier qu'on peut calculer l'angle
                if angles[idx - 1] > np.pi / 4:  # 45 degrés de changement minimum
                    sharp_turn_mask[idx] = True
                    sharp_turn_mask[idx + 1] = True  # Marquer aussi le point suivant

        return sharp_turn_mask

    def detect_symmetry_transitions(self):
        """Détecte les transitions aux plans de symétrie"""
        dot_products = np.sum(self.t_vectors[1:] * self.t_vectors[:-1], axis=1)
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))

        # Combiner critère d'angle avec position
        symmetry_mask = np.zeros(len(self.points), dtype=bool)
        potential_transitions = np.where(angles > np.pi / 4)[0] + 1

        for idx in potential_transitions:
            # Vérifier proximité avec plan de symétrie
            x, y = self.points[idx, 0:2]
            r = np.sqrt(x * x + y * y)
            theta = np.arctan2(y, x)
            sym_angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            angle_diffs = np.min(np.abs(np.mod(theta - sym_angles + np.pi, 2 * np.pi) - np.pi))
            if angle_diffs < np.radians(10):
                symmetry_mask[idx] = True

        return symmetry_mask

    def fix_extremities_vector(self):
        """Corrige les vecteurs aux points de saut selon leur position (avant/après le saut)"""
        sharp_turns = self.detect_extremities()
        indices = np.where(sharp_turns)[0]

        for i in range(0, len(indices) - 1, 2):
            start_idx = indices[i]
            end_idx = indices[i + 1]

            # Pour le point de départ: utiliser le vecteur tangent du point précédent
            if start_idx > 0:  # Vérifier qu'on n'est pas au tout premier point
                self.t_vectors[start_idx] = self.t_vectors[start_idx - 1]

            # Pour le point d'arrivée: utiliser le vecteur tangent du point suivant
            if end_idx < len(self.points) - 1:  # Vérifier qu'on n'est pas au dernier point
                self.t_vectors[end_idx] = self.t_vectors[end_idx + 1]

            # Mise à jour des normales pour les deux points
            for idx in [start_idx, end_idx]:
                self.n_vectors[idx] = np.cross(self.t_vectors[idx], self.b_vectors[idx])
                norm = np.linalg.norm(self.n_vectors[idx])
                if norm > 1e-10:
                    self.n_vectors[idx] /= norm

    def fix_symmetry_transitions(self):
        symmetry_mask = self.detect_symmetry_transitions()
        for i in range(1, len(self.points) - 1):
            if symmetry_mask[i]:
                t_prev = self.t_vectors[i - 1]
                t_next = self.t_vectors[i + 1]
                t_new = 0.5 * (t_prev - t_next)
                norm_t = np.linalg.norm(t_new)

                if norm_t > 1e-10:
                    t_new = t_new / norm_t
                else:
                    t_new = self.t_vectors[i - 1]

                self.t_vectors[i] = t_new
                n_new = np.cross(t_new, self.b_vectors[i])
                norm_n = np.linalg.norm(n_new)

                if norm_n > 1e-10:
                    self.n_vectors[i] = n_new / norm_n
                else:
                    self.n_vectors[i] = self.n_vectors[i - 1]

    def calculate_local_basis(self):
        """Vectorized local basis calculus"""
        # Initialiser les vecteurs
        tangents = np.zeros_like(self.points)
        normalized_t = np.zeros_like(self.points)

        # Calculer toutes les différences
        diffs = np.diff(self.points, axis=0)

        # Détecter les grands sauts en Z (transitions de couche)
        z_diffs = np.abs(diffs[:, 2])  # Composante Z des différences
        z_threshold = 1.0  # À ajuster selon votre géométrie
        valid_diffs = z_diffs < z_threshold

        # Assigner les tangentes seulement où les sauts en Z sont petits
        tangents[:-1][valid_diffs] = diffs[valid_diffs]
        tangents[-1] = tangents[-2]  # Dernier point

        # Normalization
        b_norms = np.linalg.norm(self.build_directions, axis=1, keepdims=True)
        tool_norms = np.linalg.norm(self.tool_directions, axis=1, keepdims=True)
        t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)

        # Masque pour éviter la division par zéro
        non_zero_mask = t_norms > 1e-10

        # Normalisation avec gestion des zéros
        normalized_b = np.zeros_like(self.build_directions)
        normalized_t = np.zeros_like(tangents)
        normalized_tool_direction = np.zeros_like(self.tool_directions)

        normalized_b[b_norms[:, 0] > 1e-10] = self.build_directions[b_norms[:, 0] > 1e-10] / b_norms[
            b_norms[:, 0] > 1e-10]
        normalized_t[non_zero_mask[:, 0]] = tangents[non_zero_mask[:, 0]] / t_norms[non_zero_mask[:, 0]]
        normalized_tool_direction[tool_norms[:, 0] > 1e-10] = self.tool_directions[tool_norms[:, 0] > 1e-10] / \
                                                              tool_norms[tool_norms[:, 0] > 1e-10]

        # Final vector product for n
        normalized_n = np.cross(normalized_t, normalized_b)

        # Normaliser n également
        n_norms = np.linalg.norm(normalized_n, axis=1, keepdims=True)
        normalized_n[n_norms[:, 0] > 1e-10] = normalized_n[n_norms[:, 0] > 1e-10] / n_norms[n_norms[:, 0] > 1e-10]

        return normalized_t, normalized_n, normalized_b, normalized_tool_direction

    def save_modified_trajectory(self, new_tool_vectors: np.ndarray, output_path: str):
        """Save a modified trajectory while keeping the original format"""
        # Copy of the original trajectory
        modified_trajectory = self.trajectory_data.copy()

        # Update tool vectors
        modified_trajectory['Tx'] = new_tool_vectors[:, 0]
        modified_trajectory['Ty'] = new_tool_vectors[:, 1]
        modified_trajectory['Tz'] = new_tool_vectors[:, 2]

        # Save as CSV
        modified_trajectory.to_csv(output_path, index=False)
        print(f"Modified trajectory saved to: {output_path}")

    @staticmethod
    def format_trajectory_data(points: np.ndarray, tool_vectors: np.ndarray,
                               build_vectors: np.ndarray) -> pd.DataFrame:
        """Create a DataFrame in the expected trajectory format"""
        trajectory = pd.DataFrame()

        # Points coordinates
        trajectory['X'] = points[:, 0]
        trajectory['Y'] = points[:, 1]
        trajectory['Z'] = points[:, 2]

        # Build vectors
        trajectory['Bx'] = build_vectors[:, 0]
        trajectory['By'] = build_vectors[:, 1]
        trajectory['Bz'] = build_vectors[:, 2]

        # Tool vectors
        trajectory['Tx'] = tool_vectors[:, 0]
        trajectory['Ty'] = tool_vectors[:, 1]
        trajectory['Tz'] = tool_vectors[:, 2]

        # Additional columns with default values
        trajectory['Extrusion'] = 1
        trajectory['WeldProcedure'] = 1
        trajectory['WeldSchedule'] = 1
        trajectory['WeaveSchedule'] = 1
        trajectory['WeaveType'] = 0

        return trajectory