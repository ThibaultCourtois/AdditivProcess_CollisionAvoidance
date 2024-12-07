import pandas as pd
import numpy as np

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

        if file_path:
            self.load_trajectory(file_path)

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