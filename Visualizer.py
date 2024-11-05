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
        self.h_layer = 1.95  # mm (constant height)
        self.w_layer = 3  # mm (constant width)
        self.L_seg = 0.32  # mm (segment length)
        self.scale_vectors = scale_vectors

        # Tool visualization parameters
        self.show_tool = False  # Toggle tool visualization
        self.tool_type = 'cylinder'  # 'cylinder' or 'stl'
        self.tool_radius = 5  # mm
        self.tool_height = 50  # mm
        self.nozzle_length = 10  # mm
        self.tool_stl_path = None  # Path to STL file
        self.current_tool_position = None  # Store clicked position

        # Setup figure and 3D axes
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Add keyboard navigation parameters
        self.current_point_index = None  # Index of current point in visible points list
        self.visible_points = None  # List of visible points
        self.visible_tools = None  # List of corresponding tool directions
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)


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
        poly = Poly3DCollection(vertices, alpha=0.3, color='gray')
        ax.add_collection3d(poly)

    #-------------------------------------------------------------------
    # Tool Visualization Methods
    #-------------------------------------------------------------------

    def set_tool_visualization(self, show_tool=True, tool_type='cylinder'):
        """
        Configure tool visualization

        Parameters:
        -----------
        show_tool : bool
            Enable/disable tool visualization
        tool_type : str
            'cylinder' or 'stl'
        """
        self.show_tool = show_tool
        self.tool_type = tool_type

    def set_tool_parameters(self, radius=None, height=None, nozzle_length=None, stl_path=None):
        """
        Set tool visualization parameters

        Parameters:
        -----------
        radius : float
            Radius for cylinder visualization
        height : float
            Height for cylinder visualization
        stl_path : str
            Path to STL file for STL visualization
        """
        if radius is not None:
            self.tool_radius = radius
        if height is not None:
            self.tool_height = height
        if nozzle_length is not None:
            self.nozzle_length = nozzle_length
        if stl_path is not None:
            self.tool_stl_path = stl_path

    def update_tool_visualization(self):
        """
        Updates the tool visualization
        """
        # Clear previous tool visualization
        if self.current_tool_position is not None:
            # Remove surface plots
            for collection in self.ax.collections:
                if hasattr(collection, 'tool_surface') and collection.tool_surface:
                    collection.remove()

            # Remove tool lines
            lines_to_remove = []
            for line in self.ax.lines:
                if hasattr(line, 'tool_line') and line.tool_line:
                    lines_to_remove.append(line)
            for line in lines_to_remove:
                line.remove()

        # Update tool visualization if enabled
        if self.show_tool and self.current_point_index is not None:
            position = self.visible_points[self.current_point_index]
            direction = self.visible_tools[self.current_point_index]

            if self.tool_type == 'cylinder':
                self.visualize_simplified_tool(position, direction)

            self.current_tool_position = position

        self.fig.canvas.draw()

    def create_tool_geometry(self, center, direction, radius=5, height=50, nozzle_length=3, n_points=32):
        """
        Create points for tool visualization with nozzle

        Parameters:
        -----------
        center : array-like
            Position of the tool tip
        direction : array-like
            Tool direction vector
        radius : float
            Radius of the cylinder
        height : float
            Height of the cylinder
        nozzle_length : float
            Length of the nozzle line
        n_points : int
            Number of points for cylinder discretization

        Returns:
        --------
        cylinder_base : ndarray
            Points forming the base of cylinder
        cylinder_top : ndarray
            Points forming the top of cylinder
        nozzle_points : ndarray
            Two points defining the nozzle line
        """
        # Normalize direction
        direction = direction / np.linalg.norm(direction)

        # Create circle points for cylinder
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(theta)
        circle_points = np.column_stack((x, y, z))

        # Create orthonormal basis with direction as z-axis
        z_axis = direction
        x_axis = np.array([1, 0, 0])
        if np.abs(np.dot(x_axis, z_axis)) > 0.9:
            x_axis = np.array([0, 1, 0])
        y_axis = np.cross(z_axis, x_axis)
        x_axis = np.cross(y_axis, z_axis)

        # Create rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

        # Transform circle points
        cylinder_base = np.dot(circle_points, rotation_matrix.T)
        cylinder_top = cylinder_base + height * direction

        # Create nozzle points (from center to nozzle_length along direction)
        nozzle_start = center
        nozzle_end = center + nozzle_length * direction
        nozzle_points = np.array([nozzle_start, nozzle_end])

        # Translate cylinder to position (center - nozzle_length * direction to place cylinder above nozzle)
        cylinder_base += (center + nozzle_length * direction)
        cylinder_top += (center + nozzle_length * direction)

        return cylinder_base, cylinder_top, nozzle_points

    def visualize_simplified_tool(self, position, direction):
        """
        Visualize tool as semi-infinite cylinder with central nozzle line
        """
        # Create tool geometry
        base_points, top_points, nozzle_points = self.create_tool_geometry(
            position, direction,
            radius=self.tool_radius,
            height=self.tool_height,
            nozzle_length=self.nozzle_length
        )

        # Plot cylinder surface
        # Using plot_surface for the cylinder wall
        theta = np.linspace(0, 2 * np.pi, len(base_points))
        z = np.linspace(0, 1, 2)
        theta, z = np.meshgrid(theta, z)

        x = np.vstack((base_points[:, 0], top_points[:, 0]))
        y = np.vstack((base_points[:, 1], top_points[:, 1]))
        z = np.vstack((base_points[:, 2], top_points[:, 2]))

        # Plot cylinder with transparency
        surf = self.ax.plot_surface(x, y, z, color='gray', alpha=0.3)
        surf.tool_surface = True  # Mark surface as belonging to tool

        # Plot cylinder edges for better visibility
        line1 = self.ax.plot(base_points[:, 0], base_points[:, 1], base_points[:, 2], 'k-', linewidth=0.5)[0]
        line2 = self.ax.plot(top_points[:, 0], top_points[:, 1], top_points[:, 2], 'k-', linewidth=0.5)[0]
        line1.tool_line = True
        line2.tool_line = True

        # Plot vertical lines at cylinder corners
        for i in range(len(base_points)):
            line = self.ax.plot([base_points[i, 0], top_points[i, 0]],
                                [base_points[i, 1], top_points[i, 1]],
                                [base_points[i, 2], top_points[i, 2]],
                                'k-', linewidth=0.5)[0]
            line.tool_line = True

        # Plot nozzle line in black
        line = self.ax.plot(nozzle_points[:, 0], nozzle_points[:, 1], nozzle_points[:, 2],
                            'k-', linewidth=2.0)[0]
        line.tool_line = True

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


        # Calculate angles for all points
        xy_angles = np.degrees(np.arctan2(display_points[:, 1], display_points[:, 0]))
        xy_angles = np.where(xy_angles < 0, xy_angles + 360, xy_angles)

        # Create angle mask
        angle_mask = xy_angles <= self.revolute_display_angle

        # Apply mask to display points
        filtered_points = display_points[angle_mask]
        filtered_b_vector = display_b_vector[angle_mask]
        filtered_tool_vector = display_tool_vector[angle_mask]

        # Calculate tangent vectors for filtered points
        tangents = np.zeros_like(filtered_points)
        tangents[:-1] = filtered_points[1:] - filtered_points[:-1]
        tangents[-1] = tangents[-2]

        # Normalize and scale vectors to fixed length
        b_vector_norm = np.linalg.norm(filtered_b_vector, axis=1)
        t_vector_norm = np.linalg.norm(tangents, axis=1)

        normalized_b_vector = filtered_b_vector / b_vector_norm[:, np.newaxis]
        normalized_t_vector = tangents / t_vector_norm[:, np.newaxis]
        normalized_n_vector = np.cross(normalized_t_vector, normalized_b_vector)

        # Plot filtered trajectory with discontinuity handling
        segments = []
        current_segment = []
        max_angle_diff = 6

        for i in range(len(display_points)):
            if angle_mask[i]:
                # First point of the segment ?
                if not current_segment:
                    current_segment.append(display_points[i])
                else:
                    # Angle calculation
                    prev_angle = np.degrees(np.arctan2(current_segment[-1][1], current_segment[-1][0]))
                    curr_angle = np.degrees(np.arctan2(display_points[i][1], display_points[i][0]))

                    # Angle normalization
                    prev_angle = prev_angle if prev_angle >= 0 else prev_angle + 360
                    curr_angle = curr_angle if curr_angle >= 0 else curr_angle + 360

                    angle_diff = min(abs(curr_angle - prev_angle),
                                     abs(curr_angle - prev_angle + 360),
                                     abs(curr_angle - prev_angle - 360))

                    # Angle condition (6°)
                    if angle_diff <= max_angle_diff:
                        current_segment.append(display_points[i])
                    else:
                        if len(current_segment) > 1:
                            segments.append(np.array(current_segment))
                        current_segment = [display_points[i]]
            elif current_segment:
                if len(current_segment) > 1:
                    segments.append(np.array(current_segment))
                current_segment = []

        # Add up last segment
        if current_segment and len(current_segment) > 1:
            segments.append(np.array(current_segment))

        # Segment plotting
        for segment_idx, segment in enumerate(segments):
            self.ax.plot(segment[:, 0],
                         segment[:, 1],
                         segment[:, 2],
                         'k-', label='Trajectory' if segment is segments[0] else "",
                         alpha=1)

        # Generate and display ellipses and vectors

            for i in range(0, len(filtered_points), self.stride):
                pt = filtered_points[i]
                t = normalized_t_vector[i]
                b = normalized_b_vector[i]
                n = normalized_n_vector[i]

                if self.ellipse_bool:
                    ellipse_points = self.construct_half_ellipse(
                        pt,
                        n,
                        b,
                        t
                    )
                    self.plot_ellipse(ellipse_points, self.ax)

                if self.vector_bool:
                    # Build direction vector
                    self.ax.quiver(pt[0], pt[1], pt[2],
                                   b[0], b[1], b[2],
                                   color='red', alpha=1,
                                   label='Build direction' if (segment_idx ==0 and i == 0) else "",
                                   normalize=True)

                    # Tangent direction vector
                    self.ax.quiver(pt[0], pt[1], pt[2],
                                   t[0], t[1], t[2],
                                   color='blue', alpha=1,
                                   label='Tangent direction' if (segment_idx ==0 and i == 0) else "",
                                   normalize=True)

                    # Normal direction vector
                    self.ax.quiver(pt[0], pt[1], pt[2],
                                   n[0], n[1], n[2],
                                   color='green', alpha=1,
                                   label='Tool direction' if (segment_idx ==0 and i == 0) else "",
                                   normalize=True)

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

    def update_visible_points(self):
        """
        Updates the list of visible trajectory points
        """
        # Get visible points (using existing angle mask)
        start_idx, end_idx = self.get_layer_points_range()
        points = self.points[start_idx:end_idx]
        tools = self.tool_directions[start_idx:end_idx]

        # Calculate angles for visibility check
        xy_angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        xy_angles = np.where(xy_angles < 0, xy_angles + 360, xy_angles)
        angle_mask = xy_angles <= self.revolute_display_angle

        self.visible_points = points[angle_mask]
        self.visible_tools = tools[angle_mask]

        # Initialiser l'index au premier point si pas encore défini
        if self.current_point_index is None and len(self.visible_points) > 0:
            self.current_point_index = 0

    #-------------------------------------------------------------------
    # Events methods
    #-------------------------------------------------------------------

    def on_key_press(self, event):
        """
        Handle keyboard events
        """
        if self.visible_points is None:
            self.update_visible_points()

        if len(self.visible_points) == 0:
            return

        # Control detection through event modifier
        ctrl_pressed = event.key.startswith('ctrl+') or (
                    hasattr(event, 'mod') and event.mod & 2)  # 2 corresponds to CTRL

        if event.key in ['4', 'ctrl+4']:  # Left equivalent
            if ctrl_pressed:
                # With Ctrl: move backward by stride points
                new_index = max(0, self.current_point_index - 10)
            else:
                # Without Ctrl: move backward by one point
                new_index = max(0, self.current_point_index - 1)

            if new_index != self.current_point_index:
                self.current_point_index = new_index
                if self.show_tool:
                    self.update_tool_visualization()

        elif event.key in ['6', 'ctrl+6']:  # Right equivalent
            if ctrl_pressed:
                # With Ctrl: move forward by stride points
                new_index = min(len(self.visible_points) - 1, self.current_point_index + 10)
            else:
                # Without Ctrl: move forward by one point
                new_index = min(len(self.visible_points) - 1, self.current_point_index + 1)

            if new_index != self.current_point_index:
                self.current_point_index = new_index
                if self.show_tool:
                    self.update_tool_visualization()

        elif event.key in ['8', 'ctrl+8']:  # Up equivalent
            self.show_tool = not self.show_tool
            self.update_tool_visualization()
    #-------------------------------------------------------------------
    # Main call
    #-------------------------------------------------------------------

    def visualize(self):
        """Main visualization method"""
        self.main_visualization()
        self.update_visible_points()
        plt.show()

