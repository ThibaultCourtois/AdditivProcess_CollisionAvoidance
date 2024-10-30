import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import Event
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from CollisionAvoidance import TrajectoryAcquisition

"""
Trajectory Visualization Class

Features:
- Visualize parts of the trajectory (quarter, half, or complete layers)
- Display t, n, b vectors at points with specified stride
- Simplified bead representation using half-ellipses with custom segmentation
- Tool visualization as infinite cylinder or using STL geometry at desired points
"""

class TrajectoryVisualizer:
    def __init__(self, trajectory_data: 'TrajectoryAcquisition', revolut_angle_display : float ,display_layers: list, stride: int,
                 ellipse_bool: bool, vector_bool: bool, scale_vectors=0.1):
        """
        Initialize the trajectory visualizer

        Parameters:
        -----------
        trajectory_data : TrajectoryAcquisition
            Data containing points and vectors of the trajectory
        display_layers : list
            List of layer indices to display
        stride : int
            Step size for vector and ellipse display
        ellipse_bool : bool
            Toggle bead geometry visualization
        vector_bool : bool
            Toggle vector visualization
        scale_vectors : float
            Scale factor for vector display
        """

        # Store trajectory data
        self.points = trajectory_data.points
        self.build_directions = trajectory_data.build_directions
        self.tool_directions = trajectory_data.tool_directions
        self.layer_indices = trajectory_data.layer_indices
        self.extrusion_data = trajectory_data.extrusion

        # Visualization control
        self.ellipse_bool = ellipse_bool
        self.vector_bool = vector_bool
        self.display_layers = display_layers
        self.stride = stride
        self.revolute_display_angle = revolut_angle_display

        # Fixed parameters from project requirements
        self.h_layer = 1.95  # mm (hauteur constante)
        self.w_layer = 3  # mm (largeur constante)
        self.L_seg = 1  # mm (longueur de segment)
        self.scale_vectors = scale_vectors

        # Setup figure and 3D axes
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')


    #-------------------------------------------------------------------
    # Ellipse Generation and Visualization Methods
    #-------------------------------------------------------------------

    def construct_half_ellipse(self, center, n_vector, b_vector, t_vector):
        """
        Create an ellipse based on a deformed rectangle

        Parameters:
        -----------
        center : array-like
            Central point for the ellipse
        n_vector : array-like
            Normal vector to the bead
        b_vector : array-like
            Build direction vector
        t_vector : array-like
            Tangent vector for segment orientation

        Returns:
        --------
        points : ndarray
            Array of 3D points forming the ellipse
        """
        seg_point_num_t = 2
        seg_point_num_n = 6

        # Points creation for the rectangle
        t_vals = np.linspace(-self.L_seg / 2, self.L_seg / 2, seg_point_num_t)
        n_vals = np.linspace(-self.w_layer / 2, self.w_layer / 2, seg_point_num_n)

        # horizontals (n = -w_layer/2, n = w_layer/2)
        points_h1 = [center + t * t_vector + (-self.w_layer / 2) * n_vector for t in t_vals]
        points_h2 = [center + t * t_vector + self.w_layer / 2 * n_vector for t in reversed(t_vals)]

        # vertical (t = L_seg/2)
        points_v1 = [center + n * n_vector + (-self.L_seg / 2 * t_vector) for n in n_vals]
        points_v2 = [center + n * n_vector + self.L_seg / 2 * t_vector for n in reversed(n_vals)]

        ellipse_points = np.concatenate((np.array(points_h1), np.array(points_v1),np.array(points_h2), np.array(points_v2)))

        # Create height deformation with a sinusoïdal function by normalizing distances between the center of the
        # rectangle and points of the rectangle between 0 and 1
        distances_to_center = np.linalg.norm(ellipse_points - center, axis=1)
        max_distance = np.max(distances_to_center)

        height_factors = np.cos(distances_to_center / max_distance * np.pi / 2)
        height_variation = (self.h_layer/2) * height_factors[:, np.newaxis] * b_vector # 3D deformation vector

        ellipse_points = ellipse_points + height_variation

        return ellipse_points

    @staticmethod
    def plot_ellipse(ellipse_points, ax):
        """
        Visualize ellipse with black outline and gray filling

        Parameters:
        -----------
        ellipse_points : ndarray
            Points defining the ellipse
        ax : Axes3D
            Matplotlib 3D axes for plotting
        """
        # Black outline creation
        # Closing the outline
        points_closed = np.append(ellipse_points, [ellipse_points[0]], axis=0)
        ax.plot(points_closed[:, 0],
                points_closed[:, 1],
                points_closed[:, 2],
                'k-', linewidth=1)

        # Filling with gray polygons
        vertices = [list(zip(points_closed[:, 0],
                             points_closed[:, 1],
                             points_closed[:, 2]))]
        poly = Poly3DCollection(vertices, alpha=0.6, color='gray')
        ax.add_collection3d(poly)

    def ellipse_display_segmentation(self, display_points, normalized_t_vector, normalized_b_vector, normalized_n_vector, extrusion):
        """
        Create new points along trajectory spaced by L_seg for ellipse display

        Parameters:
        -----------
        display_points : ndarray
            Original trajectory points
        normalized_t/b/n_vector : ndarray
            Normalized direction vectors

        Returns:
        --------
        new_points : ndarray
            New points spaced by L_seg
        new_vectors : tuple
            (t, b, n) vectors interpolated at new points
        """

        # Filtering extrusion
        last_deposit_idx = len(extrusion) - 1
        while last_deposit_idx > 0 and extrusion[last_deposit_idx] == 0:
            last_deposit_idx -= 1
        last_deposit_idx -= 1
        mask = np.zeros(len(extrusion), dtype=bool)
        mask[:last_deposit_idx + 1] = (extrusion[:last_deposit_idx + 1] == 1)

        display_points = display_points[mask]
        normalized_t_vector = normalized_t_vector[mask]
        normalized_b_vector = normalized_b_vector[mask]
        normalized_n_vector = normalized_n_vector[mask]

        # Calculate distances between points
        diff = np.diff(display_points, axis=0)
        distances = np.linalg.norm(diff, axis=1)
        cumul_dist = np.cumsum(np.insert(distances, 0, 0))  # Add 0 at start
        total_dist = cumul_dist[-1]

        # Create evenly spaced points
        num_points = int(np.ceil(total_dist / self.L_seg))
        target_distances = np.linspace(0, total_dist, num_points)

        new_points = np.zeros((num_points, 3))
        new_t = np.zeros((num_points, 3))
        new_b = np.zeros((num_points, 3))
        new_n = np.zeros((num_points, 3))

        # Interpolate points and vectors
        for i, target in enumerate(target_distances):
            # Find segment containing target distance
            idx = np.searchsorted(cumul_dist, target) - 1
            idx = max(0, min(idx, len(display_points) - 2))

            # Calculate interpolation factor
            segment_length = cumul_dist[idx + 1] - cumul_dist[idx]
            alpha = (target - cumul_dist[idx]) / segment_length if segment_length > 0 else 0

            # Interpolate point
            p0 = display_points[idx]
            p1 = display_points[idx + 1]
            new_points[i] = p0 + alpha * (p1 - p0)

            # Interpolate vectors
            new_t[i] = (1 - alpha) * normalized_t_vector[idx] + alpha * normalized_t_vector[idx + 1]
            new_b[i] = (1 - alpha) * normalized_b_vector[idx] + alpha * normalized_b_vector[idx + 1]
            new_n[i] = (1 - alpha) * normalized_n_vector[idx] + alpha * normalized_n_vector[idx + 1]

            # Renormalize vectors
            new_t[i] /= np.linalg.norm(new_t[i])
            new_b[i] /= np.linalg.norm(new_b[i])
            new_n[i] /= np.linalg.norm(new_n[i])

        return new_points, (new_t, new_b, new_n)

    #-------------------------------------------------------------------
    # Main Visualization Methods
    #-------------------------------------------------------------------
    def main_visualization(self):
        """
        Main plotting function for trajectory visualization

        Displays:
        - Complete trajectory path
        - Bead geometry (if enabled)
        - Direction vectors (if enabled)
        """
        # Get range of points to display
        start_idx, end_idx = self.get_layer_points_range()
        if start_idx == end_idx:
            print("No layers to display")
            return

        # Get points and vectors for selected layers
        display_points = self.points[start_idx:end_idx]
        display_b_vector = self.build_directions[start_idx:end_idx]
        display_tool_vector = self.tool_directions[start_idx:end_idx]

        # Plot trajectory line
        self.ax.plot(display_points[:, 0],
                     display_points[:, 1],
                     display_points[:, 2],
                     'k-', label='Trajectory', alpha=1)

        # Calculate tangent vectors
        tangents = np.zeros_like(display_points)
        tangents[:-1] = display_points[1:] - display_points[:-1]
        tangents[-1] = tangents[-2]

        # Normalize and scale vectors to fixed length
        b_vector_norm = np.linalg.norm(display_b_vector, axis=1)
        t_vector_norm = np.linalg.norm(tangents, axis=1)

        normalized_b_vector = display_b_vector / b_vector_norm[:, np.newaxis]
        normalized_t_vector = tangents / t_vector_norm[:, np.newaxis]
        normalized_n_vector = np.cross(normalized_t_vector, normalized_b_vector)

        display_ellipse_points, ellipse_vectors = self.ellipse_display_segmentation(display_points, normalized_t_vector, normalized_b_vector, normalized_n_vector, self.extrusion_data[start_idx:end_idx])

        # Draw geometry at specified stride
        for i in range(0, len(display_ellipse_points), self.stride):
            if self.ellipse_bool:
                ellipse_points = self.construct_half_ellipse(display_ellipse_points[i], ellipse_vectors[2][i], ellipse_vectors[1][i], ellipse_vectors[0][i])
                self.plot_ellipse(ellipse_points, self.ax)

        for i in range(0, len(display_points), self.stride):
            # Get local vectors
            t = normalized_t_vector[i]
            b = normalized_b_vector[i]
            n = normalized_n_vector[i]

            if self.vector_bool:
                # Build direction vector
                self.ax.quiver(display_points[i, 0],
                               display_points[i, 1],
                               display_points[i, 2],
                               b[0], b[1], b[2],
                               color='red', alpha=1,
                               label='Build direction' if i == 0 else "", normalize=True)

                # Tangent direction vector
                self.ax.quiver(display_points[i, 0],
                               display_points[i, 1],
                               display_points[i, 2],
                               t[0],
                               t[1],
                               t[2],
                               color='blue', alpha=1,
                               label='Tangent direction' if i == 0 else "", normalize=True)

                # Normal direction vector
                self.ax.quiver(display_points[i, 0],
                               display_points[i, 1],
                               display_points[i, 2],
                               n[0],
                               n[1],
                               n[2],
                               color='green', alpha=1,
                               label='Tool direction' if i == 0 else "", normalize=True)

        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        layer_str = ", ".join(str(l) for l in sorted(self.display_layers))
        self.ax.set_title(f'Trajectory with bead geometry - Layers: {layer_str}')

        # Add legend and set aspect ratio
        self.ax.legend()
        # Force equal scaling on all axes
        self.ax.set_box_aspect([1, 1, 1])

        # For graphic axe scale purposes
        x_range = np.ptp(display_points[:, 0])
        y_range = np.ptp(display_points[:, 1])
        z_range = np.ptp(display_points[:, 2])

        # Set same scales for all axes
        max_range = np.array([x_range, y_range, z_range]).max()
        mid_x = (display_points[:, 0].max() + display_points[:, 0].min()) * 0.5
        mid_y = (display_points[:, 1].max() + display_points[:, 1].min()) * 0.5
        mid_z = (display_points[:, 2].max() + display_points[:, 2].min()) * 0.5
        self.ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        self.ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        self.ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

    #-------------------------------------------------------------------
    # Utility Methods
    #-------------------------------------------------------------------

    def get_layer_points_range(self):
        """
        Calculate start and end indices for layer display

        Returns:
        --------
        tuple : (start_idx, end_idx)
            Start and end indices for the selected layers

        Raises:
        -------
        ValueError
            If layer index is out of range
        """
        if not self.display_layers:
            return 0, 0

        max_layer = max(self.display_layers)
        if max_layer >= len(self.layer_indices):
            raise ValueError(f"Layer index {max_layer} is out of range. Max layer is {len(self.layer_indices) - 1}")

        start_idx = 0
        if min(self.display_layers) > 0:
            start_idx = self.layer_indices[min(self.display_layers)]

        end_idx = self.layer_indices[max_layer]
        return start_idx, end_idx

# Fonction pour appeler la fonction de visualisation principale;
    def visualize(self):
        """Main visualization method"""
        self.main_visualization()
        plt.show()