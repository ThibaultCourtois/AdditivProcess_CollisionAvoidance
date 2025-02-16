"""
Advanced 3D Trajectory Visualization System
========================================

This script provides a comprehensive visualization system for analyzing and displaying
3D robotic trajectories, particularly focused on additive manufacturing applications.

Key Features:
------------
* Visualization of 3D printing tool paths and material deposition
* Real-time collision detection and visualization
* Interactive tool movement visualization
* Support for layer-by-layer analysis
* Advanced geometric representations including:
  - Material beads
  - Tool orientation
  - Local coordinate systems
  - Collision points

The system uses pre-calculated local bases for optimal performance and provides
various visualization options that can be toggled interactively.

Technical Details:
----------------
* Uses Matplotlib for 3D visualization
* Implements optimized geometric calculations
* Supports both high and low resolution bead visualization
* Provides keyboard controls for navigation
* Handles trajectory segmentation based on extrusion status

Usage:
-----
The system can be used to:
1. Analyze robotic trajectories for potential collisions
2. Visualize material deposition patterns
3. Verify tool orientations and movements
4. Debug path planning issues
5. Validate layer-by-layer build strategies
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

np.set_printoptions(threshold=np.inf)
np.seterr(all='ignore')  # Ignore numpy warnings for improved performance

"""
Optimized visualization system using pre-calculated local bases.
This module provides a comprehensive set of tools for 3D visualization 
of robotic trajectories, including tool paths, beads, and collision detection.
"""

# --------------------------------
# Local Basis Management
# --------------------------------

class LocalBasisManager:
    """
    Encapsulates access to basis vectors from TrajectoryDataManager.
    Manages the storage and retrieval of local coordinate systems (tangent, normal, build direction)
    for each point along the trajectory.
    """

    def __init__(self, trajectory_manager=None):
        """
        Initialize the local basis manager.

        Args:
            trajectory_manager: Optional TrajectoryDataManager instance for vector data
        """
        self.trajectory_manager = trajectory_manager
        self.update_from_manager()

    def update_from_manager(self):
        """
        Retrieves vector data from the TrajectoryDataManager.
        Updates internal vector storage with the latest data from the trajectory manager.
        """
        if self.trajectory_manager:
            self.t_vectors = self.trajectory_manager.t_vectors  # Tangent vectors
            self.n_vectors = self.trajectory_manager.n_vectors  # Normal vectors
            self.b_vectors = self.trajectory_manager.b_vectors  # Build direction vectors
            self.tool_vectors = self.trajectory_manager.tool_directions  # Tool orientation vectors

    def get_local_basis(self, index):
        """
        Retrieves the local basis vectors for a given point.

        Args:
            index: Index of the point in the trajectory

        Returns:
            tuple: (tangent, normal, build, tool_direction) vectors at the specified index
        """
        return (self.t_vectors[index],
                self.n_vectors[index],
                self.b_vectors[index],
                self.tool_vectors[index])

# --------------------------------
# Geometric Visualization Classes
# --------------------------------

class GeometryVisualizer:
    """
    Handles the geometric visualization of trajectory elements including beads and tools.
    Provides optimized methods for generating and rendering 3D geometry.
    """

    def __init__(self, basis_manager):
        """
        Initialize the geometry visualizer.

        Args:
            basis_manager: LocalBasisManager instance for accessing coordinate systems
        """
        self.basis_manager = basis_manager

    def generate_bead_geometry_for_segment(self, segment_points, t_vectors, n_vectors, b_vectors, bead_width, bead_height,
                                           low_res_bead_bool):
        """
        Generates optimized bead geometry with adaptive length for a trajectory segment.

        Args:
            segment_points: Array of points defining the trajectory segment
            t_vectors: Tangent vectors at each point
            n_vectors: Normal vectors at each point
            b_vectors: Build direction vectors at each point
            bead_width: Width of the deposited material layer
            bead_height: Height of the deposited material layer
            low_res_bead_bool: Flag for using low resolution geometry

        Returns:
            List of section points arrays defining the bead geometry
        """
        segment_length = len(segment_points)
        if segment_length < 2:
            return None

        # Adjust number of points based on resolution setting
        n_section_points = 4 if low_res_bead_bool else 8

        # Generate half-ellipse for each point in segment
        all_sections = []

        for i in range(segment_length):
            point = segment_points[i]
            # Normalize local coordinate system vectors
            t = t_vectors[i] / np.linalg.norm(t_vectors[i])
            n = n_vectors[i] / np.linalg.norm(n_vectors[i])
            b = b_vectors[i] / np.linalg.norm(b_vectors[i])

            # Calculate adaptive length based on adjacent points
            prev_dist = np.linalg.norm(segment_points[i] - segment_points[i - 1]) if i > 0 else 0
            next_dist = np.linalg.norm(segment_points[i] - segment_points[i + 1]) if i < segment_length - 1 else 0

            # Compute adaptive segment length with bounds
            segment_length = np.mean([d for d in [prev_dist, next_dist] if d > 0])
            segment_length = np.clip(segment_length, 0.2, 1.0)  # Min/max limits in mm

            # Generate base rectangle points (optimized)
            n_vals = np.linspace(-bead_width / 2, bead_width / 2, n_section_points)
            section_points = []

            # Optimized point construction
            # Front points
            for n_val in n_vals:
                section_points.append(point + n_val * n + segment_length / 2 * t)
            # Back points (reverse order)
            for n_val in reversed(n_vals):
                section_points.append(point + n_val * n - segment_length / 2 * t)

            section_points = np.array(section_points)

            # Half-ellipse deformation (optimized calculation)
            distances = np.abs(section_points - point)
            max_distance = np.max(np.linalg.norm(distances, axis=1))
            height_factors = np.cos(np.linalg.norm(distances, axis=1) / max_distance * np.pi / 2)
            height_variation = (bead_height / 2) * height_factors[:, np.newaxis] * b

            section = section_points + height_variation
            all_sections.append(section)

        return all_sections

    def plot_bead_sections(self, ax, sections):
        """
        Renders bead sections with proper vertex management.

        Args:
            ax: Matplotlib 3D axis for rendering
            sections: List of section points arrays to render
        """
        aluminum_color = '#D3D3D3'

        for section in sections:
            n_points = len(section)
            mid_point = n_points // 2

            # Optimized surface creation
            for i in range(mid_point - 1):
                # Create quadrilateral with 4 points in proper format
                quad = [
                    section[i].tolist(),
                    section[i + 1].tolist(),
                    section[-(i + 2)].tolist(),
                    section[-(i + 1)].tolist(),
                ]

                # Create collection with single quad
                poly = Poly3DCollection([quad])
                poly.set_facecolor(aluminum_color)
                poly.set_edgecolor('black')
                poly.set_linewidth(0.1)
                poly.set_rasterized(True)
                poly.set_zsort('min')
                ax.add_collection3d(poly)

    def generate_tool_geometry(self, point, tool_dir):
        """
        Generates robust 3D tool geometry.

        Args:
            point: Starting point for tool placement
            tool_dir: Tool direction vector

        Returns:
            Dictionary containing tool geometry components:
            - base: Base circle points
            - top: Top circle points
            - nozzle_start: Start point of nozzle
            - nozzle_end: End point of nozzle
        """
        # Normalize direction vector
        tool_dir = tool_dir / np.linalg.norm(tool_dir)

        # Build robust orthonormal basis
        z_axis = tool_dir
        if abs(z_axis[0]) < 1e-6 and abs(z_axis[1]) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0]))
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Generate cylinder points
        theta = np.linspace(0, 2 * np.pi, self.tool_discretization_points)
        circle_points = np.column_stack((
            self.tool_radius * np.cos(theta),
            self.tool_radius * np.sin(theta),
            np.zeros_like(theta)
        ))

        # Apply transformation
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        nozzle_end = point + self.nozzle_length * tool_dir

        # Create base and top circles
        base_circle = np.dot(circle_points, rotation_matrix.T) + nozzle_end
        top_circle = base_circle + self.tool_height * tool_dir

        return {
            'base': base_circle,
            'top': top_circle,
            'nozzle_start': point,
            'nozzle_end': nozzle_end
        }

    def set_parameters(self, bead_width=None, bead_height=None,
                       tool_radius=None, tool_length=None, bead_discretization_points=None,
                       nozzle_length=None, tool_discretization_points=None):
        """
        Configure geometric parameters for visualization.

        Args:
            bead_width: Width of deposited material bead
            bead_height: Height of deposited material bead
            tool_radius: Radius of the tool
            tool_length: Length of the tool
            bead_discretization_points: Number of points to use in bead cross-sections
            nozzle_length: Length of the tool nozzle
            tool_discretization_points: Number of points to use in tool circular sections
        """
        # Tool parameters
        if tool_radius:
            self.tool_radius = tool_radius
        if tool_length:
            self.tool_height = tool_length
        if bead_width:
            self.bead_width = bead_width
        if bead_height:
            self.bead_height = bead_height
        if bead_discretization_points:
            self.bead_discretization_points = bead_discretization_points
        if nozzle_length:
            self.nozzle_length = nozzle_length
        if tool_discretization_points:
            self.tool_discretization_points = tool_discretization_points

# --------------------------------
# Advanced Trajectory Visualization
# --------------------------------

class AdvancedTrajectoryVisualizer:
    """
    Main visualization manager for robotic trajectories.
    Handles the display of trajectories, tools, beads, and collision detection results.
    """

    def __init__(self, trajectory_manager=None, collision_manager=None, display_layers=None,
                 revolut_angle=360, stride=1):
        """
        Initialize the advanced trajectory visualizer.

        Args:
            trajectory_manager: Manager for trajectory data
            collision_manager: Manager for collision detection
            display_layers: List of layers to display
            revolut_angle: Maximum angle for revolution display (degrees)
            stride: Step size for vector display
        """
        self.trajectory_manager = trajectory_manager
        self.collision_manager = collision_manager
        self.basis_manager = LocalBasisManager(trajectory_manager)
        self.geometry_visualizer = GeometryVisualizer(self.basis_manager)
        self.collision_points = None
        if collision_manager:
            self.collision_points = collision_manager.detect_collisions_optimized()

        # Visualization parameters
        self.display_layers = display_layers or []
        self.revolut_angle = revolut_angle
        self.stride = stride

        # Visualization states
        self.show_beads = False
        self.show_tool = False
        self.show_vectors = False
        self.show_collisions = False
        self.low_res_bead = True
        self.show_collision_candidates = False
        self.show_collision_bases = False

        # Navigation control
        self.current_point = 0
        self.visible_points = None
        self.visible_indices = None

        # Matplotlib figure
        self.fig = None
        self.ax = None
        self.tool_artists = []

    def setup_visualization(self, show_beads=False, low_res_bead=True, show_tool=False,
                            show_vectors=False, show_collisions=False, show_collision_candidates=False,
                            show_collision_bases=False):
        """
        Configure visualization options.

        Args:
            show_beads: Toggle bead visualization
            low_res_bead: Use low resolution for bead geometry
            show_tool: Toggle tool visualization
            show_vectors: Toggle vector visualization
            show_collisions: Toggle collision point visualization
            show_collision_candidates: Toggle collision candidate visualization
            show_collision_bases: Toggle collision basis visualization
        """
        self.show_beads = show_beads
        self.show_tool = show_tool
        self.show_vectors = show_vectors
        self.show_collisions = show_collisions
        self.low_res_bead = low_res_bead
        self.show_collision_candidates = show_collision_candidates
        self.show_collision_bases = show_collision_bases

    def apply_layer_filter(self, layers=None, angle_limit=None):
        """
        Filter points considering extrusion status and angle limits.

        Args:
            layers: List of layer indices to display
            angle_limit: Maximum angle for point filtering
        """
        if not layers:
            return

        # Get start and end indices
        start_idx = self.trajectory_manager.layer_indices[min(layers)]
        end_idx = (self.trajectory_manager.layer_indices[max(layers) + 1]
                   if max(layers) + 1 < len(self.trajectory_manager.layer_indices)
                   else len(self.trajectory_manager.points))

        # Select points and vectors
        selected_points = self.trajectory_manager.points[start_idx:end_idx]
        selected_t = self.basis_manager.t_vectors[start_idx:end_idx]
        selected_n = self.basis_manager.n_vectors[start_idx:end_idx]
        selected_b = self.basis_manager.b_vectors[start_idx:end_idx]
        selected_tool = self.trajectory_manager.tool_directions[start_idx:end_idx]
        selected_extrusion = self.trajectory_manager.extrusion[start_idx:end_idx]

        # Apply angle filtering
        if angle_limit is not None and angle_limit != 360:
            angles = np.degrees(np.arctan2(selected_points[:, 1], selected_points[:, 0]))
            angles = np.where(angles < 0, angles + 360, angles)
            angle_mask = angles <= angle_limit

            selected_points = selected_points[angle_mask]
            selected_t = selected_t[angle_mask]
            selected_n = selected_n[angle_mask]
            selected_b = selected_b[angle_mask]
            selected_tool = selected_tool[angle_mask]
            selected_extrusion = selected_extrusion[angle_mask]

            global_indices = np.arange(start_idx, end_idx)[angle_mask]
        else:
            global_indices = np.arange(start_idx, end_idx)

        # Store filtered data
        self.visible_points = selected_points
        self.visible_t_vector = selected_t
        self.visible_n_vector = selected_n
        self.visible_b_vector = selected_b
        self.visible_tool_vector = selected_tool
        self.visible_extrusion = selected_extrusion

        # Create mapping between global and visible indices
        self.global_to_visible_indices = {global_idx: local_idx
                                          for local_idx, global_idx in enumerate(global_indices)}
        self.visible_indices = global_indices

    def create_figure(self):
        """
        Create and configure matplotlib figure with advanced optimizations.
        Sets up the 3D visualization environment with optimal rendering settings.
        """
        self.fig = plt.figure(figsize=(12, 8), dpi=100)  # Reduced resolution for performance
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Rendering optimizations
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Disable background panels
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Disable grid for performance
        self.ax.grid(False)

        # Disable automatic limit calculation
        self.ax.autoscale(enable=False)

        # Use orthographic projection (faster)
        self.ax.set_proj_type('ortho')

        # Disable anti-aliasing
        self.fig.set_facecolor('white')
        self.ax.set_facecolor('white')

        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self.handle_keyboard)

    def visualize_trajectory(self):
        """
        Visualize trajectory with extrusion-based segmentation.
        Handles the main visualization process including trajectory segments,
        beads, tools, and collision information.
        """
        if not self.fig:
            self.create_figure()

        # Apply layer and angle filters
        self.apply_layer_filter(self.display_layers, self.revolut_angle)

        # Segment based on extrusion
        points = self.visible_points
        segments = []
        current_segment = []

        # Get start index for filtering
        start_idx = self.trajectory_manager.layer_indices[min(self.display_layers)]

        # Create segments based on extrusion points
        for i in range(len(points)):
            if self.visible_extrusion[i] == 1:  # Point with extrusion
                if len(current_segment) == 0 or self.visible_extrusion[i - 1] == 1:
                    current_segment.append(points[i])
                else:
                    # New segment if coming from non-extrusion point
                    if len(current_segment) > 1:
                        segments.append(np.array(current_segment))
                    current_segment = [points[i]]
            else:
                # Close current segment if exists
                if len(current_segment) > 1:
                    segments.append(np.array(current_segment))
                current_segment = []

        # Add final segment
        if len(current_segment) > 1:
            segments.append(np.array(current_segment))

        # Plot trajectory segments
        for i, segment in enumerate(segments):
            self.ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                         'k-', label='Trajectory' if i == 0 else "")

        # Add optional visualization elements
        if self.show_vectors:
            self._draw_vectors()
        if self.show_beads:
            self._draw_beads()
        if self.show_collisions:
            self._draw_collisions()
        if self.show_tool:
            self.update_tool_visualization()
        if self.show_collision_candidates:
            self.visualize_collision_candidates(
                self.visible_points,
                self.visible_n_vector,
                self.visible_b_vector
            )
        if self.show_collision_bases:
            self._draw_collision_bases()

        # Setup plot limits and aspect
        self._setup_plot_limits()

    def update_tool_visualization(self):
        """
        Update tool visualization with robust geometric representation.
        Manages the display of the tool's 3D geometry, including the nozzle and body.
        """
        # Clear previous tool visualization
        for artist in self.tool_artists:
            artist.remove()
        self.tool_artists.clear()

        if not self.show_tool or self.current_point is None:
            return

        # Get geometric data
        point = self.visible_points[self.current_point]
        tool_dir = self.visible_tool_vector[self.current_point]

        # Generate tool geometry
        geom = self.geometry_visualizer.generate_tool_geometry(point, tool_dir)

        # Create cylinder representation
        for circle in [geom['base'], geom['top']]:
            # Close the circle for continuous display
            circle_closed = np.vstack([circle, circle[0]])
            line = self.ax.plot(circle_closed[:, 0],
                                circle_closed[:, 1],
                                circle_closed[:, 2],
                                'k-', linewidth=0.5)[0]
            self.tool_artists.append(line)

        # Create cylinder surface
        cylinder_x = np.vstack((geom['base'][:, 0], geom['top'][:, 0]))
        cylinder_y = np.vstack((geom['base'][:, 1], geom['top'][:, 1]))
        cylinder_z = np.vstack((geom['base'][:, 2], geom['top'][:, 2]))

        surf = self.ax.plot_surface(cylinder_x, cylinder_y, cylinder_z,
                                    color='gray', alpha=0.3)
        self.tool_artists.append(surf)

        # Draw nozzle line
        line = self.ax.plot([geom['nozzle_start'][0], geom['nozzle_end'][0]],
                            [geom['nozzle_start'][1], geom['nozzle_end'][1]],
                            [geom['nozzle_start'][2], geom['nozzle_end'][2]],
                            'k-', linewidth=2.0)[0]
        self.tool_artists.append(line)

        self.fig.canvas.draw()

    def _draw_beads(self):
        """
        Draw material beads using a segmented approach.
        Creates 3D geometry for each material deposition segment, considering
        extrusion status and local coordinate systems.
        """
        if len(self.visible_indices) < 1:
            return

        # Get bead parameters from collision manager
        bead_width = self.collision_manager.collision_candidates_generator.bead_width
        bead_height = self.collision_manager.collision_candidates_generator.bead_height

        # Initialize trajectory segmentation
        points = self.visible_points
        segments = []
        current_segment = []
        current_t = []
        current_n = []
        current_b = []

        # Process points for segment creation
        for i in range(len(points)):
            if self.visible_extrusion[i] == 1:
                if len(current_segment) == 0 or self.visible_extrusion[i - 1] == 1:
                    # Add to current segment
                    current_segment.append(points[i])
                    current_t.append(self.visible_t_vector[i])
                    current_n.append(self.visible_n_vector[i])
                    current_b.append(self.visible_b_vector[i])
                else:
                    # Create new segment with collected data
                    if len(current_segment) > 1:
                        segments.append({
                            'points': np.array(current_segment),
                            't_vectors': np.array(current_t),
                            'n_vectors': np.array(current_n),
                            'b_vectors': np.array(current_b)
                        })
                    # Initialize new segment with current point
                    current_segment = [points[i]]
                    current_t = [self.visible_t_vector[i]]
                    current_n = [self.visible_n_vector[i]]
                    current_b = [self.visible_b_vector[i]]
            else:
                # Close current segment if it exists
                if len(current_segment) > 1:
                    segments.append({
                        'points': np.array(current_segment),
                        't_vectors': np.array(current_t),
                        'n_vectors': np.array(current_n),
                        'b_vectors': np.array(current_b)
                    })
                # Reset segment collections
                current_segment = []
                current_t = []
                current_n = []
                current_b = []

        # Add final segment if necessary
        if len(current_segment) > 1:
            segments.append({
                'points': np.array(current_segment),
                't_vectors': np.array(current_t),
                'n_vectors': np.array(current_n),
                'b_vectors': np.array(current_b)
            })

        # Generate and render bead geometry for each segment
        for segment in segments:
            sections = self.geometry_visualizer.generate_bead_geometry_for_segment(
                segment['points'],
                segment['t_vectors'],
                segment['n_vectors'],
                segment['b_vectors'],
                bead_width,
                bead_height,
                self.low_res_bead
            )
            if sections:
                self.geometry_visualizer.plot_bead_sections(self.ax, sections)

    def _draw_vectors(self):
        """
        Draw local coordinate system vectors with enhanced tool vector handling.
        Displays tangent, normal, and build direction vectors along with tool vectors
        when they significantly differ from the build direction.
        """
        plot_points = []
        plot_vectors = []
        colors = []

        for i in range(0, len(self.visible_points), self.stride):
            point = self.visible_points[i]
            t = self.visible_t_vector[i]
            n = self.visible_n_vector[i]
            b = self.visible_b_vector[i]
            tool = self.visible_tool_vector[i]

            # Add base vectors to visualization
            plot_points.extend([point, point, point])
            plot_vectors.extend([t, n, b])
            colors.extend(['blue', 'green', 'red'])

            # Handle tool vector with enhanced angle checking
            norm_b = b / np.linalg.norm(b)
            norm_tool = tool / np.linalg.norm(tool)
            angle_diff = np.arccos(np.clip(np.dot(norm_b, norm_tool), -1.0, 1.0))

            # Only display tool vector if angle difference is significant (>5 degrees)
            if np.degrees(angle_diff) > 5:
                plot_points.append(point)
                plot_vectors.append(tool)
                colors.append('orange')

        # Vectorized plotting
        plot_points = np.array(plot_points)
        plot_vectors = np.array(plot_vectors)

        if len(plot_points) > 0:
            self.ax.quiver(
                plot_points[:, 0], plot_points[:, 1], plot_points[:, 2],
                plot_vectors[:, 0], plot_vectors[:, 1], plot_vectors[:, 2],
                colors=colors, normalize=True
            )

            # Create legend with dynamic tool vector inclusion
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', label='Tangent direction'),
                Line2D([0], [0], color='green', label='Normal direction'),
                Line2D([0], [0], color='red', label='Build direction')
            ]
            if 'orange' in colors:
                legend_elements.append(Line2D([0], [0], color='orange', label='Tool direction'))

            self.ax.legend(handles=legend_elements)

    def _draw_collisions(self):
        """
        Draw collision points with optimized marker size.
        Displays collision points as small red markers for better visibility
        while maintaining performance.
        """
        if self.collision_manager and self.collision_points is not None:
            # Filter to show only collisions within visible points
            collision_indices = np.where(self.collision_points)[0]
            mask = np.isin(collision_indices, self.visible_indices)

            if np.any(mask):
                collision_points = self.trajectory_manager.points[collision_indices[mask]]
                self.ax.scatter(collision_points[:, 0],
                                collision_points[:, 1],
                                collision_points[:, 2],
                                c='red', marker='x', s=3,  # Reduced marker size
                                label='Collisions')

    def _draw_collision_bases(self):
        """
        Draw local coordinate systems at collision points.
        Visualizes the tangent, normal, and build direction vectors at points
        where collisions were detected to aid in understanding and resolving issues.
        """
        if self.collision_manager and self.collision_points is not None:
            # Get global indices of collision points
            collision_indices = self.collision_manager.get_collision_indices()

            # Map to visible indices
            visible_collision_indices = [
                self.global_to_visible_indices[global_idx]
                for global_idx in collision_indices
                if global_idx in self.global_to_visible_indices
            ]

            if not visible_collision_indices:
                print("No collision points visible after filtering.")
                return

            plot_points = []
            plot_vectors = []
            colors = []

            for idx in visible_collision_indices:
                point = self.visible_points[idx]
                t = self.visible_t_vector[idx]
                n = self.visible_n_vector[idx]
                b = self.visible_b_vector[idx]

                # Normalize vectors for visualization
                t = t / np.linalg.norm(t)
                n = n / np.linalg.norm(n)
                b = b / np.linalg.norm(b)

                # Add base vectors
                plot_points.extend([point, point, point])
                plot_vectors.extend([t, n, b])
                colors.extend(['blue', 'green', 'red'])

            if len(plot_points) > 0:
                plot_points = np.array(plot_points)
                plot_vectors = np.array(plot_vectors)

                self.ax.quiver(
                    plot_points[:, 0], plot_points[:, 1], plot_points[:, 2],
                    plot_vectors[:, 0], plot_vectors[:, 1], plot_vectors[:, 2],
                    colors=colors, normalize=True,
                )

            # Create legend with vector descriptions
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', label='Tangent direction (t)'),
                Line2D([0], [0], color='green', label='Normal direction (n)'),
                Line2D([0], [0], color='red', label='Build direction (b)'),
            ]
            self.ax.legend(handles=legend_elements)
            print(f"Visualizing local bases for {len(visible_collision_indices)} collision points.")

    def _setup_plot_limits(self):
        """
        Configure plot display limits and aspect ratio.
        Calculates optimal view boundaries based on trajectory extent
        and ensures consistent scaling across all axes.
        """
        if len(self.visible_points) > 0:
            # Calculate dimensions
            x_range = np.ptp(self.visible_points[:, 0])
            y_range = np.ptp(self.visible_points[:, 1])
            z_range = np.ptp(self.visible_points[:, 2])

            max_range = max(x_range, y_range, z_range)
            center = np.mean(self.visible_points, axis=0)

            # Apply symmetric limits around center
            self.ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
            self.ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
            self.ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)

            # Force equal aspect ratio for proper 3D visualization
            self.ax.set_box_aspect([1, 1, 1])

    def visualize_collision_candidates(self, points, normal_vectors, build_vectors):
        """
        Visualize potential collision points using filtered data.
        Displays points where collisions might occur based on geometry and tool path.

        Args:
            points: Array of trajectory points
            normal_vectors: Array of normal vectors at each point
            build_vectors: Array of build direction vectors at each point
        """
        if len(points) == 0:
            return

        # Get dimensions from collision manager
        bead_width = self.collision_manager.collision_candidates_generator.bead_width
        bead_height = self.collision_manager.collision_candidates_generator.bead_height

        # Calculate candidate points with stride
        strided_points = points[::self.stride]
        strided_normal = normal_vectors[::self.stride]
        strided_build = build_vectors[::self.stride]

        # Calculate left and right candidate points
        left_points = strided_points + (bead_width / 2) * strided_normal
        right_points = strided_points - (bead_width / 2) * strided_normal

        # Display lateral candidate points
        all_points = np.vstack([left_points, right_points])
        self.ax.scatter(all_points[:, 0],
                        all_points[:, 1],
                        all_points[:, 2],
                        color='darkcyan',
                        marker='o',
                        s=1,
                        label='Collision candidates')

        # Calculate and display top points for last layer
        if self.display_layers:
            max_layer = max(self.display_layers)
            start_idx = self.trajectory_manager.layer_indices[max_layer - 1] if max_layer > 0 else 0
            end_idx = self.trajectory_manager.layer_indices[max_layer]

            # Get points for last layer
            is_last_layer = np.arange(len(strided_points))[::self.stride]
            last_layer_points = strided_points[is_last_layer]
            last_layer_build = strided_build[is_last_layer]
            top_points = last_layer_points + bead_height * last_layer_build

            # Display top points
            self.ax.scatter(top_points[:, 0],
                            top_points[:, 1],
                            top_points[:, 2],
                            color='darkcyan',
                            marker='o',
                            s=1)

    def handle_keyboard(self, event):
        """
        Handle keyboard input for interactive visualization control.

        Supported keys:
        - Left/4: Move tool back one point
        - Right/6: Move tool forward one point
        - t/8: Toggle tool visibility
        - Ctrl+Left/4: Move tool back 10 points
        - Ctrl+Right/6: Move tool forward 10 points

        Args:
            event: Matplotlib keyboard event
        """
        if not hasattr(self, 'current_point'):
            self.current_point = 0

        # Basic movement controls
        if event.key in ['left', '4']:
            self.current_point = max(0, self.current_point - 1)
            self.update_tool_visualization()
        elif event.key in ['right', '6']:
            self.current_point = min(len(self.visible_points) - 1, self.current_point + 1)
            self.update_tool_visualization()
        elif event.key in ['t', '8']:
            self.show_tool = not self.show_tool
            self.update_tool_visualization()

        # Extended movement controls with modifier
        elif event.key.startswith('ctrl+'):
            step = 10
            if 'left' in event.key or '4' in event.key:
                self.current_point = max(0, self.current_point - step)
            elif 'right' in event.key or '6' in event.key:
                self.current_point = min(len(self.visible_points) - 1, self.current_point + step)
            self.update_tool_visualization()

    def update_visualization(self):
        """
        Update visualization after state changes.
        Refreshes the display with current visualization settings.
        """
        self.ax.clear()
        self.visualize_trajectory()
        plt.draw()

    def show(self):
        """
        Display the visualization.
        Creates the figure if not already created and shows the plot.
        """
        if not self.fig:
            self.create_figure()
        self.visualize_trajectory()
        plt.show()