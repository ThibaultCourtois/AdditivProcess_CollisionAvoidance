import pandas as pd
import numpy as np
import trimesh


class TrajectoryAcquisition:
    def __init__(self, file_path):
        # Lecture du fichier CSV
        self.trajectory_data = pd.read_csv(file_path)

        # Extraction des différents vecteurs en arrays numpy
        self.points = self.trajectory_data[['X', 'Y', 'Z']].values
        self.build_directions = self.trajectory_data[['Bx', 'By', 'Bz']].values
        self.tool_directions = self.trajectory_data[['Tx', 'Ty', 'Tz']].values
        self.extrusion = self.trajectory_data['Extrusion'].values
        # Identification des couches basée sur la colonne Extrusion
        self.layer_indices = self.identify_layers()

    def identify_layers(self):
        """Identifie les indices de début de chaque couche"""
        # Une nouvelle couche commence quand Extrusion passe de 0 à 1
        extrusion_col = self.trajectory_data['Extrusion'].values
        layer_starts_indices = np.where(extrusion_col == 0)[0] + 1
        return layer_starts_indices

    def get_layer_points(self, layer_index):
        """Retourne les points d'une couche spécifique"""
        start_idx = self.layer_indices[layer_index]
        end_idx = self.layer_indices[layer_index + 1] if layer_index + 1 < len(self.layer_indices) else len(self.points)
        return self.points[start_idx:end_idx]

    def get_layer_build_directions(self, layer_index):
        """Retourne les directions de construction d'une couche spécifique"""
        start_idx = self.layer_indices[layer_index]
        end_idx = self.layer_indices[layer_index + 1] if layer_index + 1 < len(self.layer_indices) else len(self.points)
        return self.build_directions[start_idx:end_idx]


class CollisionAvoidance:
    def __init__(self, trajectory, bead_width, bead_height, tool_radius, tool_length, tool_geometry_path):
        self.trajectory = trajectory
        self.bead_width = bead_width
        self.bead_height = bead_height
        self.tool_radius = tool_radius
        self.tool_length = tool_length
        self.modified_trajectory = []
        self.tool_orientations = []
        self.tool_mesh = trimesh.load_mesh(tool_geometry_path)
