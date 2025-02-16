"""
TrajectoryManager: Class for managing and processing WAAM manufacturing trajectories.

This class handles:
- Loading and saving trajectory data
- Computing local coordinate bases
- Managing layer information
- Processing trajectory modifications
- Handling vector calculations and normalizations
"""

import pandas as pd
import numpy as np

class TrajectoryManager:
    """
    Manages trajectory data and operations for Wire Arc Additive Manufacturing.

    Handles trajectory data loading, processing, and manipulation, including:
    - Local coordinate system calculations
    - Layer identification and management
    - Vector calculations and normalizations
    - Trajectory modifications and saving
    """

    def __init__(self, file_path=None):
        """
        Initialize TrajectoryManager with optional trajectory file.

        Args:
            file_path (str, optional): Path to trajectory CSV file
        """
        # Main trajectory data structures
        self.trajectory_data = None  # Complete trajectory DataFrame
        self.points = None  # Point coordinates (Nx3 array)
        self.build_directions = None  # Build direction vectors
        self.tool_directions = None  # Tool direction vectors
        self.extrusion = None  # Extrusion state flags
        self.layer_indices = None  # Starting indices of layers

        # Local coordinate system vectors
        self.t_vectors = None  # Tangent vectors
        self.n_vectors = None  # Normal vectors
        self.b_vectors = None  # Binormal vectors
        self.tool_vectors = None  # Tool orientation vectors

        if file_path:
            self.load_trajectory(file_path)
            self.compute_and_store_local_bases()

    def compute_and_store_local_bases(self):
        """
        Calculate and store local coordinate bases for all trajectory points.

        Computes orthogonal local bases (t,n,b) for each point where:
        - t: Tangent vector along trajectory
        - n: Normal vector perpendicular to trajectory
        - b: Build direction vector
        """
        # Calculate initial orthogonal basis
        t_vec, n_vec, b_vec, tool_vec = self.calculate_local_basis()

        # Store computed vectors
        self.t_vectors = t_vec
        self.n_vectors = n_vec
        self.b_vectors = b_vec
        self.tool_vectors = tool_vec

        # Correct vectors at trajectory discontinuities
        self.correct_boundary_vectors()

    def load_trajectory(self, file_path):
        """
        Load trajectory data from CSV file and extract key components.

        Args:
            file_path (str): Path to trajectory CSV file containing:
                - Point coordinates (X,Y,Z)
                - Build directions (Bx,By,Bz)
                - Tool directions (Tx,Ty,Tz)
                - Extrusion flags
        """
        # Load CSV data into DataFrame
        self.trajectory_data = pd.read_csv(file_path)

        # Extract vector components
        self.points = self.trajectory_data[['X', 'Y', 'Z']].values
        self.build_directions = self.trajectory_data[['Bx', 'By', 'Bz']].values
        self.tool_directions = self.trajectory_data[['Tx', 'Ty', 'Tz']].values
        self.extrusion = self.trajectory_data['Extrusion'].values

        # Identify layer boundaries
        self.layer_indices = self.identify_layers()

    def identify_layers(self):
        """
        Identify starting indices of each manufacturing layer.

        Layers are identified by transitions in the extrusion state
        where extrusion changes from 0 to 1.

        Returns:
            np.array: Array of layer starting indices
        """
        extrusion_col = self.trajectory_data['Extrusion'].values
        layer_starts_indices = np.where(extrusion_col == 0)[0]

        # Ensure first layer starts at index 0
        if len(layer_starts_indices) == 0 or layer_starts_indices[0] != 0:
            layer_starts_indices = np.insert(layer_starts_indices, 0, 0)

        return layer_starts_indices

    def get_layer_points(self, layer_index):
        """
        Get point coordinates for a specific layer.

        Args:
            layer_index (int): Index of the desired layer

        Returns:
            np.array: Array of point coordinates for the specified layer
        """
        start_idx = self.layer_indices[layer_index]
        end_idx = (self.layer_indices[layer_index + 1]
                   if layer_index + 1 < len(self.layer_indices)
                   else len(self.points))
        return self.points[start_idx:end_idx]

    def get_layer_build_directions(self, layer_index):
        """Returning build directions of a specific layer"""
        start_idx = self.layer_indices[layer_index]
        end_idx = self.layer_indices[layer_index + 1] if layer_index + 1 < len(self.layer_indices) else len(self.points)
        return self.build_directions[start_idx:end_idx]

    def detect_jumps_from_extrusion(self):
        """
        Detect trajectory discontinuities using extrusion state changes.

        Creates a boolean mask identifying points where:
        - Extrusion starts (0 to 1 transition)
        - Extrusion ends (1 to 0 transition)

        Returns:
            np.array: Boolean mask marking transition points
        """
        # Detect changes in extrusion state
        extrusion_changes = np.diff(self.extrusion)
        jump_indices = np.where(extrusion_changes != 0)[0]

        # Create mask for affected points
        jump_mask = np.zeros(len(self.points), dtype=bool)

        # Mark points before and after each extrusion change
        for idx in jump_indices:
            jump_mask[idx] = True  # Point before transition
            jump_mask[idx + 1] = True  # Point after transition

        return jump_mask

    def correct_boundary_vectors(self):
        """
        Correct vector orientations at trajectory discontinuities.

        Ensures smooth vector transitions at:
        - End points of extrusion segments
        - Start points of new extrusion segments

        This prevents sudden changes in orientation that could affect
        manufacturing quality.
        """
        # Find extrusion state transitions
        extrusion_changes = np.where(np.diff(self.extrusion))[0]

        for idx in extrusion_changes:
            # Handle end of extrusion segment (1 to 0 transition)
            if self.extrusion[idx] == 1:
                # Use previous point's vector for last extrusion point
                if idx > 0 and self.extrusion[idx - 1] == 1:
                    self.t_vectors[idx] = self.t_vectors[idx - 1]
                    # Recalculate normal vector to maintain orthogonality
                    self.n_vectors[idx] = np.cross(self.t_vectors[idx], self.b_vectors[idx])
                    norm = np.linalg.norm(self.n_vectors[idx])
                    if norm > 1e-10:
                        self.n_vectors[idx] /= norm

            # Handle start of extrusion segment (0 to 1 transition)
            elif idx + 1 < len(self.extrusion) and self.extrusion[idx + 1] == 1:
                # Use next point's vector for first extrusion point
                if idx + 2 < len(self.extrusion) and self.extrusion[idx + 2] == 1:
                    self.t_vectors[idx + 1] = self.t_vectors[idx + 2]
                    # Recalculate normal vector to maintain orthogonality
                    self.n_vectors[idx + 1] = np.cross(self.t_vectors[idx + 1], self.b_vectors[idx + 1])
                    norm = np.linalg.norm(self.n_vectors[idx + 1])
                    if norm > 1e-10:
                        self.n_vectors[idx + 1] /= norm

    def calculate_local_basis(self):
        """
        Calculate orthonormal local basis vectors for each trajectory point.

        Implements a vectorized calculation of (t,n,b) basis where:
        - t: Tangent vector computed from trajectory direction
        - b: Build direction vector (given in input data)
        - n: Normal vector computed as cross product of t and b

        Handles special cases:
        - Non-extrusion points
        - Trajectory endpoints
        - Vector propagation across discontinuities

        Returns:
            tuple: (normalized_t, normalized_n, normalized_b, normalized_tool_direction)
                Each element is a Nx3 array of normalized vectors
        """
        # Initialize output arrays
        tangents = np.zeros_like(self.points)
        normalized_t = np.zeros_like(self.points)
        normalized_b = np.zeros_like(self.build_directions)
        normalized_tool_direction = np.zeros_like(self.tool_directions)

        # Calculate trajectory differences for tangent computation
        diffs = np.diff(self.points, axis=0)

        # Create mask for extrusion points
        extrusion_mask = self.extrusion == 1

        # Compute initial tangents for extrusion points
        tangents[:-1][extrusion_mask[:-1]] = diffs[extrusion_mask[:-1]]
        if extrusion_mask[-1]:
            tangents[-1] = tangents[-2]  # Handle last point

        # Propagate tangents through non-extrusion regions
        last_valid_tangent = None
        for i in range(len(self.points)):
            if extrusion_mask[i]:
                # Use calculated tangent for extrusion points
                if np.all(tangents[i] == 0):  # Handle endpoint
                    tangents[i] = (last_valid_tangent
                                   if last_valid_tangent is not None
                                   else diffs[-1])
                last_valid_tangent = tangents[i]
            else:
                # Handle non-extrusion points
                if last_valid_tangent is not None:
                    tangents[i] = last_valid_tangent
                elif i < len(diffs):  # Look ahead for next valid tangent
                    next_extrusion_idx = i + 1
                    while (next_extrusion_idx < len(self.points)
                           and not extrusion_mask[next_extrusion_idx]):
                        next_extrusion_idx += 1
                    if next_extrusion_idx < len(self.points):
                        tangents[i] = tangents[next_extrusion_idx]

        # Normalize all vectors using broadcasting
        for vectors, normalized in [
            (tangents, normalized_t),
            (self.build_directions, normalized_b),
            (self.tool_directions, normalized_tool_direction)
        ]:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            mask = norms > 1e-10
            normalized[mask[:, 0]] = vectors[mask[:, 0]] / norms[mask[:, 0]]

        # Calculate normal vectors through cross product
        normalized_n = np.cross(normalized_t, normalized_b)
        n_norms = np.linalg.norm(normalized_n, axis=1, keepdims=True)
        mask = n_norms > 1e-10
        normalized_n[mask[:, 0]] = normalized_n[mask[:, 0]] / n_norms[mask[:, 0]]

        return normalized_t, normalized_n, normalized_b, normalized_tool_direction

    def save_modified_trajectory(self, new_tool_vectors: np.ndarray, output_path: str):
        """
        Save trajectory with modified tool vectors while preserving original format.

        Args:
            new_tool_vectors (np.ndarray): Updated tool orientation vectors (Nx3 array)
            output_path (str): Path where the modified trajectory will be saved

        The saved trajectory maintains all original data except for tool vectors,
        ensuring compatibility with manufacturing systems.
        """
        # Create a deep copy to avoid modifying original data
        modified_trajectory = self.trajectory_data.copy()

        # Update tool vector components
        modified_trajectory['Tx'] = new_tool_vectors[:, 0]
        modified_trajectory['Ty'] = new_tool_vectors[:, 1]
        modified_trajectory['Tz'] = new_tool_vectors[:, 2]

        # Save to CSV, maintaining original format
        modified_trajectory.to_csv(output_path, index=False)
        print(f"Modified trajectory saved to: {output_path}")

    def compute_average_build_vectors(self):
        """
        Calculate average build direction vectors for each layer.

        This method:
        1. Extracts build vectors for each layer
        2. Computes the mean vector per layer
        3. Normalizes the resulting vectors

        Returns:
            np.ndarray: Array of normalized average build vectors for each layer
        """
        avg_build_vectors = []

        # Process each layer
        for i in range(len(self.layer_indices)):
            # Get layer boundaries
            start_idx = self.layer_indices[i]
            end_idx = (self.layer_indices[i + 1]
                       if i + 1 < len(self.layer_indices)
                       else len(self.points))

            # Extract and average build vectors for current layer
            layer_build_vectors = self.b_vectors[start_idx:end_idx]
            avg_vector = np.mean(layer_build_vectors, axis=0)

            # Normalize the average vector
            norm = np.linalg.norm(avg_vector)
            if norm > 1e-10:
                avg_vector /= norm

            avg_build_vectors.append(avg_vector)

        return np.array(avg_build_vectors)

    @staticmethod
    def format_trajectory_data(points: np.ndarray, tool_vectors: np.ndarray,
                               build_vectors: np.ndarray) -> pd.DataFrame:
        """
        Create a properly formatted trajectory DataFrame from component arrays.

        Args:
            points (np.ndarray): Point coordinates (Nx3 array)
            tool_vectors (np.ndarray): Tool orientation vectors (Nx3 array)
            build_vectors (np.ndarray): Build direction vectors (Nx3 array)

        Returns:
            pd.DataFrame: Formatted trajectory data with all required columns:
                - Point coordinates (X, Y, Z)
                - Build vectors (Bx, By, Bz)
                - Tool vectors (Tx, Ty, Tz)
                - Manufacturing parameters (Extrusion, WeldProcedure, etc.)
        """
        trajectory = pd.DataFrame()

        # Add point coordinates
        trajectory['X'] = points[:, 0]
        trajectory['Y'] = points[:, 1]
        trajectory['Z'] = points[:, 2]

        # Add build direction vectors
        trajectory['Bx'] = build_vectors[:, 0]
        trajectory['By'] = build_vectors[:, 1]
        trajectory['Bz'] = build_vectors[:, 2]

        # Add tool orientation vectors
        trajectory['Tx'] = tool_vectors[:, 0]
        trajectory['Ty'] = tool_vectors[:, 1]
        trajectory['Tz'] = tool_vectors[:, 2]

        # Add manufacturing parameters with default values
        trajectory['Extrusion'] = 1  # Enable material deposition
        trajectory['WeldProcedure'] = 1  # Default welding procedure
        trajectory['WeldSchedule'] = 1  # Default schedule
        trajectory['WeaveSchedule'] = 1  # Default weave pattern
        trajectory['WeaveType'] = 0  # No weaving

        return trajectory