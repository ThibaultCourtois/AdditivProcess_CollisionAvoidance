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
        layer_starts_indices = np.where(extrusion_col == 0)[0]

        # Ajouter 0 comme premier index si nécessaire
        if len(layer_starts_indices) == 0 or layer_starts_indices[0] != 0:
            layer_starts_indices = np.insert(layer_starts_indices, 0, 0)

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

    def detect_jumps_from_extrusion(self):
        """Détecte les points de saut en utilisant l'information d'extrusion"""
        # Les points de saut sont là où l'extrusion change
        extrusion_changes = np.diff(self.extrusion)
        jump_indices = np.where(extrusion_changes != 0)[0]

        # Créer un masque pour les points concernés
        jump_mask = np.zeros(len(self.points), dtype=bool)

        # Marquer les points avant et après chaque changement d'extrusion
        for idx in jump_indices:
            jump_mask[idx] = True
            jump_mask[idx + 1] = True

        return jump_mask

    def fix_extremities_vector(self):
        """Corrige les vecteurs aux points de saut en utilisant l'information d'extrusion"""
        jump_mask = self.detect_jumps_from_extrusion()
        indices = np.where(jump_mask)[0]

        for i in range(0, len(indices) - 1, 2):
            start_idx = indices[i]
            end_idx = indices[i + 1]

            # Si l'extrusion s'arrête, utiliser le vecteur suivant
            if self.extrusion[start_idx] == 1 and self.extrusion[start_idx + 1] == 0:
                if end_idx + 1 < len(self.points):
                    self.t_vectors[start_idx] = self.t_vectors[end_idx + 1]
                    self.t_vectors[end_idx] = self.t_vectors[end_idx + 1]
            # Si l'extrusion démarre, utiliser le vecteur précédent
            else:
                if start_idx > 0:
                    self.t_vectors[start_idx] = self.t_vectors[start_idx - 1]
                    self.t_vectors[end_idx] = self.t_vectors[start_idx - 1]

            # Mise à jour des normales pour les deux points
            for idx in [start_idx, end_idx]:
                self.n_vectors[idx] = np.cross(self.t_vectors[idx], self.b_vectors[idx])
                norm = np.linalg.norm(self.n_vectors[idx])
                if norm > 1e-10:
                    self.n_vectors[idx] /= norm

    def calculate_local_basis(self):
        """Vectorized local basis calculus without Z filtering"""
        # Initialiser les vecteurs
        tangents = np.zeros_like(self.points)
        normalized_t = np.zeros_like(self.points)

        # Calculer toutes les différences
        diffs = np.diff(self.points, axis=0)

        # Assigner les tangentes pour tous les points
        tangents[:-1] = diffs
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

    def compute_average_build_vectors(self):
        """Calcule les vecteurs build moyens par couche"""
        avg_build_vectors = []

        for i in range(len(self.layer_indices)):
            start_idx = self.layer_indices[i]
            end_idx = (self.layer_indices[i + 1]
                       if i + 1 < len(self.layer_indices)
                       else len(self.points))

            # Extraire les vecteurs build de la couche
            layer_build_vectors = self.b_vectors[start_idx:end_idx]

            # Calculer la moyenne
            avg_vector = np.mean(layer_build_vectors, axis=0)

            # Normaliser
            norm = np.linalg.norm(avg_vector)
            if norm > 1e-10:
                avg_vector /= norm

            avg_build_vectors.append(avg_vector)

        return np.array(avg_build_vectors)

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