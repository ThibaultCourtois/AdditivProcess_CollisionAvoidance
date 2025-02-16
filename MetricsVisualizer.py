"""
MetricsVisualizer: A class for visualizing collision avoidance metrics and corrections.

This class provides various visualization methods for analyzing tool angle corrections
and collision avoidance results in Wire Arc Additive Manufacturing (WAAM).
It includes methods for:
- Layer-by-layer tilt angle analysis
- Threshold-based correction visualization
- Polar visualization of angle corrections
- Heat map generation
- 3D visualization of corrections
"""

import matplotlib.pyplot as plt
import numpy as np


class MetricsVisualizer:
    """
    Visualization tool for analyzing collision avoidance metrics and corrections.

    This class provides multiple visualization methods to analyze the results
    of collision avoidance algorithms and tool angle corrections.
    """

    def __init__(self, collision_manager, trajectory_manager):
        """
        Initialize the metrics visualizer.

        Args:
            collision_manager: Manager handling collision detection and resolution
            trajectory_manager: Manager handling trajectory data and modifications
        """
        self.collision_avoidance = collision_manager
        self.trajectory_manager = trajectory_manager

    def visualize_tilt_angles_by_layer(self, selected_layers=None, threshold=5):
        """
        Visualize correction angles for each layer with threshold detection.

        Creates a plot showing tilt angle corrections per layer, with optional
        filtering for specific layers and detection of sudden angle changes.

        Args:
            selected_layers (list, optional): Specific layer indices to display.
                If None, shows all layers.
            threshold (float): Angle threshold in degrees for detecting sudden changes.
                Changes larger than this value are marked in red.
        """
        # Get correction data
        tilt_angles = self.collision_avoidance.get_correction_angles()
        problematic_points = self.collision_avoidance.get_problematic_points()
        # Map points to their respective layers
        layer_indices = self.trajectory_manager.layer_indices
        num_layers = len(layer_indices)
        layer_dict = {i: [] for i in range(num_layers)}

        # Organize corrections by layer
        for i, point_idx in enumerate(problematic_points):
            for layer in range(num_layers):
                if point_idx < layer_indices[layer]:
                    layer_dict[layer - 1].append((point_idx, tilt_angles[i]))
                    break
            else:
                layer_dict[num_layers - 1].append((point_idx, tilt_angles[i]))

        # Create visualization
        cmap = plt.get_cmap("tab20")
        plt.figure(figsize=(12, 6))

        # Plot data for each layer
        for layer, data in layer_dict.items():
            if not data or (selected_layers and layer not in selected_layers):
                continue

            indices, angles = zip(*data)

            # Detect sudden changes in correction angles
            sudden_changes = np.where(np.abs(np.diff(angles)) > threshold)[0]

            # Plot main correction angles
            plt.plot(indices, angles, marker='o', linestyle='-', markersize=3,
                     alpha=0.7, color=cmap(layer % 20), label=f'Layer {layer + 1}')

            # Mark sudden changes
            if len(sudden_changes) > 0:
                plt.scatter(np.array(indices)[sudden_changes],
                            np.array(angles)[sudden_changes],
                            color='red', marker='x', s=50,
                            label=f'Sudden changes (Layer {layer + 1})')

        # Add plot elements
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel('Point Index')
        plt.ylabel('Correction Angle (°)')
        plt.title(f'Correction Angles by Layer (Changes > {threshold}° marked)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def visualize_tilt_angle_with_threshold(self, collision_manager, threshold=5):
        """
        Visualize correction angles with threshold highlighting.

        Creates a plot showing all correction angles with highlighted regions
        where angles exceed the specified threshold.

        Args:
            collision_manager: Manager containing correction data
            threshold (float): Angle threshold in degrees for highlighting
        """
        tilt_angles = np.array(collision_manager.get_correction_angles())
        problematic_points = np.array(collision_manager.get_problematic_points())

        plt.figure(figsize=(12, 6))
        plt.plot(problematic_points, tilt_angles, label='Correction angles',
                 color='dodgerblue', alpha=0.7)

        # Highlight regions exceeding threshold
        too_large = np.abs(tilt_angles) > threshold
        plt.fill_between(problematic_points, tilt_angles, where=too_large,
                         color='red', alpha=0.3, label=f'Angle > {threshold}°')

        plt.xlabel("Point Index")
        plt.ylabel("Correction Angle (°)")
        plt.title(f"Correction Angles with {threshold}° Threshold")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def visualize_tilt_angles_polar(self, threshold=10):
        """
        Create a polar plot of correction angles.

        Displays correction angles in a polar coordinate system, with radius
        representing point sequence and angle representing the correction magnitude.

        Args:
            threshold (float): Threshold in degrees for marking sudden changes
        """
        # Convert correction angles to radians
        tilt_angles = np.radians(self.collision_avoidance.get_correction_angles())
        problematic_points = self.collision_avoidance.get_problematic_points()

        num_points = len(problematic_points)
        radii = np.arange(num_points)  # Increasing radius for point separation

        # Detect sudden changes in correction angles
        sudden_changes = np.where(np.abs(np.diff(tilt_angles)) > np.radians(threshold))[0] + 1

        # Create polar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        ax.set_theta_zero_location("N")  # 0° at top
        ax.set_theta_direction(-1)  # Counter-clockwise

        # Plot regular corrections
        ax.scatter(tilt_angles, radii, c='blue', s=10, alpha=0.7,
                   label="Normal corrections")

        # Mark sudden changes
        if len(sudden_changes) > 0:
            ax.scatter(tilt_angles[sudden_changes], radii[sudden_changes],
                       c='red', s=30, marker='x',
                       label=f"Sudden changes (> {threshold}°)")

        ax.set_rticks([])
        ax.set_title("Angular Map of Corrections", fontsize=14, fontweight='bold')
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.show()

    def visualize_tilt_heatmap(self, collision_manager, trajectory_manager):
        """
        Create a heatmap visualization of correction angles by layer.

        Generates a 2D heatmap where:
        - X-axis represents point indices
        - Y-axis represents layer numbers
        - Color intensity represents correction angle magnitude

        Args:
            collision_manager: Manager containing correction data
            trajectory_manager: Manager containing trajectory and layer information
        """
        # Extract correction data
        tilt_angles = np.array(collision_manager.get_correction_angles())
        problematic_points = np.array(collision_manager.get_problematic_points())

        # Map points to their respective layers
        layer_indices = trajectory_manager.layer_indices
        num_layers = len(layer_indices)
        layers = np.zeros_like(problematic_points)

        # Determine layer for each point
        for i, point_idx in enumerate(problematic_points):
            for layer in range(num_layers):
                if point_idx < layer_indices[layer]:
                    layers[i] = layer - 1
                    break
            else:
                layers[i] = num_layers - 1

        # Create heatmap visualization
        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(problematic_points, layers, c=tilt_angles,
                              cmap='coolwarm', alpha=0.75, edgecolors='k')

        plt.colorbar(scatter, label="Correction Angle (°)")
        plt.xlabel("Point Index")
        plt.ylabel("Layer Index")
        plt.title("Heatmap of Correction Angles by Layer")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def visualize_tilt_angle_variation(self, collision_manager):
        """
        Visualize the variation in correction angles between consecutive points.

        Creates a plot showing how correction angles change from point to point,
        helping identify regions of rapid angle changes.

        Args:
            collision_manager: Manager containing correction data
        """
        # Get correction data
        tilt_angles = np.array(collision_manager.get_correction_angles())
        problematic_points = np.array(collision_manager.get_problematic_points())

        # Calculate point-to-point variations
        variations = np.diff(tilt_angles)

        # Create variation plot
        plt.figure(figsize=(12, 6))
        plt.plot(problematic_points[:-1], variations, marker='o', linestyle='-',
                 markersize=3, color='purple', alpha=0.7)

        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Point Index")
        plt.ylabel("Angle Variation (°)")
        plt.title("Point-to-Point Variation in Correction Angles")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def visualize_tilt_angles_3D(self, collision_manager, trajectory_manager):
        """
        Create a 3D visualization of correction angles across layers.

        Generates a 3D scatter plot where:
        - X-axis: Point indices
        - Y-axis: Layer numbers
        - Z-axis: Correction angles
        - Color: Magnitude of correction

        Args:
            collision_manager: Manager containing correction data
            trajectory_manager: Manager containing trajectory and layer information
        """
        # Extract correction data
        tilt_angles = np.array(collision_manager.get_correction_angles())
        problematic_points = np.array(collision_manager.get_problematic_points())

        # Map points to layers
        layer_indices = trajectory_manager.layer_indices
        num_layers = len(layer_indices)
        layers = np.zeros_like(problematic_points)

        # Determine layer for each point
        for i, point_idx in enumerate(problematic_points):
            for layer in range(num_layers):
                if point_idx < layer_indices[layer]:
                    layers[i] = layer - 1
                    break
            else:
                layers[i] = num_layers - 1

        # Create 3D visualization
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(problematic_points, layers, tilt_angles, c=tilt_angles,
                   cmap='coolwarm', alpha=0.75)

        ax.set_xlabel("Point Index")
        ax.set_ylabel("Layer Index")
        ax.set_zlabel("Correction Angle (°)")
        ax.set_title("3D View of Correction Angles")
        plt.show()

    def visualize_corrected_points_3D(self, collision_manager, trajectory_manager):
        """
        Create a 3D visualization of corrected points in physical space.

        Generates a 3D scatter plot showing the actual positions of corrected points
        in the manufacturing space, with colors indicating correction magnitudes.

        Args:
            collision_manager: Manager containing correction data
            trajectory_manager: Manager containing trajectory point coordinates
        """
        # Get correction data and corresponding points
        tilt_angles = np.array(collision_manager.get_correction_angles())
        problematic_points = np.array(collision_manager.get_problematic_points())
        corrected_points = trajectory_manager.points[problematic_points]

        # Create 3D visualization
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot points with color-coded correction angles
        scatter = ax.scatter(corrected_points[:, 0],  # X coordinates
                             corrected_points[:, 1],  # Y coordinates
                             corrected_points[:, 2],  # Z coordinates
                             c=tilt_angles,  # Color by correction angle
                             cmap='coolwarm',  # Color scheme
                             s=50,  # Point size
                             alpha=0.7)  # Transparency

        # Set labels and title
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("3D Position of Corrected Points")

        # Add colorbar showing angle scale
        plt.colorbar(scatter, label="Correction Angle (°)")
        plt.show()

        

