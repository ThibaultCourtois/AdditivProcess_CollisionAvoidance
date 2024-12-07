import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


"""
Trajectory Visualization Class

Features:
- Visualize parts of the trajectory (quarter, half, or complete layers)
- Display t, n, b vectors at points with specified stride
- Simplified bead representation using half-ellipses with custom segmentation
- Tool visualization as infinite cylinder or using STL geometry at desired points
"""

class TrajectoryVisualizer:
    def __init__(self, trajectory_manager, collision_manager, revolut_angle_display, display_layers, stride,
                 ellipse_bool, vector_bool, show_collision_candidates_bool,
                 show_collision_candidates_segments_bool, show_problematic_trajectory_points_bool,
                 collision_points=None, scale_vectors=0.1):
        # Existing initialization code remains the same
        self.trajectory_data = trajectory_manager
        self.points = np.asarray(trajectory_manager.points)
        self.build_directions = np.asarray(trajectory_manager.build_directions)
        self.tool_directions = np.asarray(trajectory_manager.tool_directions)
        self.layer_indices = trajectory_manager.layer_indices
        self.extrusion_data = trajectory_manager.extrusion

        # Visualization control
        self.ellipse_bool = ellipse_bool
        self.vector_bool = vector_bool
        self.display_layers = display_layers
        self.stride = stride
        self.revolute_display_angle = revolut_angle_display
        self.angle_mask = None

        # Store collision points directly
        self.show_problematic_trajectory_points = show_problematic_trajectory_points_bool
        self.collision_points = collision_points
        self.collision_manager = collision_manager


        # Fixed parameters from project requirements
        self.h_layer = collision_manager.collision_candidates_generator.h_layer
        self.w_layer = collision_manager.collision_candidates_generator.w_layer
        self.L_seg = 0.32  # mm (segment length)
        self.scale_vectors = scale_vectors

        # Tool visualization parameters
        self.show_tool = False
        self.tool_type = 'cylinder'
        self.tool_radius = 5
        self.tool_height = 50
        self.nozzle_length = 10
        self.current_tool_position = None

        # Calculus points visualization
        self.show_collision_candidates = show_collision_candidates_bool
        self.show_collision_candidates_segments = show_collision_candidates_segments_bool

        # Setup figure and 3D axes
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Add keyboard navigation parameters
        self.current_point_index = None
        self.visible_points = None
        self.visible_vectors = None
        self.visible_layers = None

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.collision_manager = collision_manager

    #-------------------------------------------------------------------
    # Calculus function callback
    #-------------------------------------------------------------------

    def detect_collisions(self):
        """Now only detects collisions if no collision points were provided"""
        if self.collision_points is None and self.show_problematic_trajectory_points:
            self.collision_points = self.collision_manager.detect_initial_collisions()
        return self.collision_points



    #-------------------------------------------------------------------
    # Ellipse Generation and Visualization Methods
    #-------------------------------------------------------------------
    def plot_collision_points(self, points, normal_vectors, build_vectors, is_last_layer_mask):
        """Vectorized collision points plotting"""
        # Calculate all points at once
        left_points = points + (self.w_layer / 2) * normal_vectors
        right_points = points - (self.w_layer / 2) * normal_vectors

        # Plot all points in one call
        all_points = np.vstack([left_points, right_points])
        self.ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
                        color='darkcyan', marker='o', s=1)

        if np.any(is_last_layer_mask):
            # Handle last layer points only where needed
            last_layer_points = points[is_last_layer_mask]
            last_layer_build = build_vectors[is_last_layer_mask]
            top_points = last_layer_points + self.h_layer * last_layer_build

            self.ax.scatter(top_points[:, 0], top_points[:, 1], top_points[:, 2],
                            color='darkcyan', marker='o', s=1)

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

    def set_tool_parameters(self, radius, height, nozzle_length):
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
            direction = self.visible_tool_vector[self.current_point_index]
            self.visualize_simplified_tool(position, direction)
            self.current_tool_position = position
        self.fig.canvas.draw()

    def create_tool_geometry(self, center, direction, radius, height, nozzle_length, n_points):
        """
        Create points for tool visualization with nozzle with constant radius
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

        # Choose x_axis vector to avoid singularities
        if abs(z_axis[0]) < 1e-6 and abs(z_axis[1]) < 1e-6:
            # If z_axis is close to (0,0,±1)
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            # Use cross product with (0,0,1) to get perpendicular vector
            x_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0]))

        # Normalize x_axis
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Get y_axis through cross product
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Recalculate x_axis to ensure orthogonality
        x_axis = np.cross(y_axis, z_axis)

        # Create rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

        # Transform circle points
        cylinder_base = np.dot(circle_points, rotation_matrix.T)
        cylinder_top = cylinder_base + height * direction

        # Create nozzle points
        nozzle_start = center
        nozzle_end = center + nozzle_length * direction
        nozzle_points = np.array([nozzle_start, nozzle_end])

        # Translate cylinder to position
        cylinder_base += (center + nozzle_length * direction)
        cylinder_top += (center + nozzle_length * direction)

        return cylinder_base, cylinder_top, nozzle_points

    def visualize_simplified_tool(self, position, direction):
        """
        Visualize tool as simple cylinder with nozzle line
        """
        # Create tool geometry
        base_points, top_points, nozzle_points = self.create_tool_geometry(
            position, direction,
            radius=self.tool_radius,
            height=self.tool_height,
            nozzle_length=self.nozzle_length
        )

        # Plot cylinder surface - simplest way but keeping radius constant
        n_points = len(base_points)
        cylinder_x = np.vstack((base_points[:, 0], top_points[:, 0]))
        cylinder_y = np.vstack((base_points[:, 1], top_points[:, 1]))
        cylinder_z = np.vstack((base_points[:, 2], top_points[:, 2]))

        # Plot cylinder with transparency
        surf = self.ax.plot_surface(cylinder_x, cylinder_y, cylinder_z, color='gray', alpha=0.3)
        surf.tool_surface = True

        # Plot base and top circles
        line1 = self.ax.plot(np.append(base_points[:, 0], base_points[0, 0]),
                             np.append(base_points[:, 1], base_points[0, 1]),
                             np.append(base_points[:, 2], base_points[0, 2]),
                             'k-', linewidth=0.5)[0]
        line2 = self.ax.plot(np.append(top_points[:, 0], top_points[0, 0]),
                             np.append(top_points[:, 1], top_points[0, 1]),
                             np.append(top_points[:, 2], top_points[0, 2]),
                             'k-', linewidth=0.5)[0]
        line1.tool_line = True
        line2.tool_line = True

        # Plot nozzle line
        line = self.ax.plot(nozzle_points[:, 0], nozzle_points[:, 1], nozzle_points[:, 2],
                            'k-', linewidth=2.0)[0]
        line.tool_line = True

    #-------------------------------------------------------------------
    # Main Visualization Methods
    #-------------------------------------------------------------------
    def main_visualization(self):
        """
        Main plotting function for trajectory visualization
        """
        if self.show_problematic_trajectory_points:
            if self.collision_points is None:
                self.detect_collisions()

        # Get points and vectors for selected layers
        self.update_visible_points_vectors()

        # Get the layer indices
        start_idx, end_idx = self.get_layer_points_range()

        # Get the last displayed layer index and its ending point index
        max_displayed_layer = max(self.display_layers)
        last_layer_start_idx = self.layer_indices[max_displayed_layer - 1] if max_displayed_layer > 0 else 0
        last_layer_end_idx = self.layer_indices[max_displayed_layer]

        # Plot filtered trajectory with discontinuity handling
        segments = []
        current_segment = []
        max_angle_diff = 6

        for i in range(len(self.points)):
            if self.angle_mask[i]:
                # First point of the segment?
                if not current_segment:
                    current_segment.append(self.points[i])
                else:
                    # Calculate angles between consecutive points
                    prev_angle = np.degrees(np.arctan2(current_segment[-1][1], current_segment[-1][0]))
                    curr_angle = np.degrees(np.arctan2(self.points[i][1], self.points[i][0]))

                    # Normalize angles to [0, 360] range
                    prev_angle = prev_angle if prev_angle >= 0 else prev_angle + 360
                    curr_angle = curr_angle if curr_angle >= 0 else curr_angle + 360

                    angle_diff = min(abs(curr_angle - prev_angle),
                                     abs(curr_angle - prev_angle + 360),
                                     abs(curr_angle - prev_angle - 360))

                    if angle_diff <= max_angle_diff:
                        current_segment.append(self.points[i])
                    else:
                        if len(current_segment) > 1:
                            segments.append(np.array(current_segment))
                        current_segment = [self.points[i]]
            elif current_segment:
                # Close current segment if outside angle mask
                if len(current_segment) > 1:
                    segments.append(np.array(current_segment))
                current_segment = []

        # Add final segment if needed
        if current_segment and len(current_segment) > 1:
            segments.append(np.array(current_segment))

        # Plot trajectory segments
        for segment_idx, segment in enumerate(segments):
            self.ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                         'k-', label='Trajectory' if segment is segments[0] else "",
                         alpha=1)

        # === OPTIONAL VISUALIZATION ELEMENTS ===
        if self.vector_bool or self.ellipse_bool or self.show_collision_candidates:
            # Calculate tangent vectors
            tangents = np.zeros_like(self.visible_points)
            tangents[:-1] = self.visible_points[1:] - self.visible_points[:-1]
            tangents[-1] = tangents[-2]

            # Calculate normalized basis vectors
            b_vector_norm = np.linalg.norm(self.visible_b_vector, axis=1)
            t_vector_norm = np.linalg.norm(tangents, axis=1)

            normalized_b_vector = self.visible_b_vector / b_vector_norm[:, np.newaxis]
            normalized_t_vector = tangents / t_vector_norm[:, np.newaxis]
            normalized_n_vector = np.cross(normalized_t_vector, normalized_b_vector)

            # Display optional elements based on flags
            if self.vector_bool:
                if self.vector_bool:
                    self.plot_direction_vectors(
                        points=self.visible_points[::self.stride],
                        t_vectors=normalized_t_vector[::self.stride],
                        n_vectors=normalized_n_vector[::self.stride],
                        b_vectors=normalized_b_vector[::self.stride],
                        tool_vectors=self.visible_tool_vector[::self.stride]
                    )

            if self.ellipse_bool:
                # Plot ellipse geometry for each point
                for i in range(0, len(self.visible_points), self.stride):
                    ellipse_points = self.construct_half_ellipse(
                        self.visible_points[i],
                        normalized_n_vector[i],
                        normalized_b_vector[i],
                        normalized_t_vector[i]
                    )
                    self.plot_ellipse(ellipse_points, self.ax)

            if self.show_collision_candidates:
                # Plot collision points for debugging
                for i in range(0, len(self.visible_points), self.stride):
                    total_idx = start_idx + i
                    is_last_layer = (total_idx >= last_layer_start_idx) and (total_idx < last_layer_end_idx)
                    self.plot_collision_points(
                        self.visible_points[i],
                        normalized_n_vector[i],
                        normalized_b_vector[i],
                        is_last_layer
                    )

        # After plotting the trajectory but before vectors/ellipses
        if self.show_problematic_trajectory_points and self.collision_points is not None:
            # Get indices for the selected layers
            start_idx, end_idx = self.get_layer_points_range()

            # Get all collision indices
            all_collision_indices = np.where(self.collision_points)[0]

            # Filter only the collisions within our layer range
            layer_mask = (all_collision_indices >= start_idx) & (all_collision_indices < end_idx)
            layer_collision_indices = all_collision_indices[layer_mask]

            # Apply angle filtering
            collision_points = self.trajectory_data.points[layer_collision_indices]
            collision_angles = np.degrees(np.arctan2(collision_points[:, 1], collision_points[:, 0]))
            collision_angles = np.where(collision_angles < 0, collision_angles + 360, collision_angles)
            angle_mask = collision_angles <= self.revolute_display_angle

            # Filter collision points by angle
            final_collision_points = collision_points[angle_mask]

            print(f"Total number of collision points: {np.sum(self.collision_points)}")
            print(f"Number of collision points in selected layers: {len(layer_collision_indices)}")
            print(f"Number of collision points after angle filtering: {len(final_collision_points)}")

            if len(final_collision_points) > 0:
                self.ax.scatter(
                    final_collision_points[:, 0],
                    final_collision_points[:, 1],
                    final_collision_points[:, 2],
                    color='red',
                    s=1,
                    marker='o',
                    label='Problematic trajectory points',
                    zorder=5
                )

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
        x_range = np.ptp(self.visible_points[:, 0])
        y_range = np.ptp(self.visible_points[:, 1])
        z_range = np.ptp(self.visible_points[:, 2])

        # Set same scales for all axes
        max_range = np.array([x_range, y_range, z_range]).max()
        mid_x = (self.visible_points[:, 0].max() + self.visible_points[:, 0].min()) * 0.5
        mid_y = (self.visible_points[:, 1].max() + self.visible_points[:, 1].min()) * 0.5
        mid_z = (self.visible_points[:, 2].max() + self.visible_points[:, 2].min()) * 0.5
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

    def update_visible_points_vectors(self):
        """Updates visible points with stride applied early"""
        start_idx, end_idx = self.get_layer_points_range()

        if start_idx == end_idx:
            print("No layers to display")
            return

        # Apply stride early in the process
        indices = np.arange(start_idx, end_idx)
        points = self.points[indices]
        b_vector = self.build_directions[indices]
        tool_vector = self.tool_directions[indices]

        # Calculate angles for visibility check
        xy_angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        xy_angles = np.where(xy_angles < 0, xy_angles + 360, xy_angles)
        angle_mask = xy_angles <= self.revolute_display_angle

        self.points = points
        self.angle_mask = angle_mask
        self.visible_points = points[angle_mask][::self.stride]
        self.visible_b_vector = b_vector[angle_mask][::self.stride]
        self.visible_tool_vector = tool_vector[angle_mask][::self.stride]

        if self.current_point_index is None and len(self.visible_points) > 0:
            self.current_point_index = 0

    def plot_direction_vectors(self, points, t_vectors, n_vectors, b_vectors, tool_vectors):
        """Optimized vector plotting with optional tool vectors"""
        # Initialize lists for plotting
        plot_points = []
        plot_vectors = []
        colors = []
        labels = []

        # Add base vectors
        for i in range(len(points)):
            # Add point and vectors for local basis
            plot_points.extend([points[i], points[i], points[i]])
            plot_vectors.extend([b_vectors[i], t_vectors[i], n_vectors[i]])
            colors.extend(['red', 'blue', 'green'])
            if i == 0:  # Add labels only for the first point
                labels.extend(['Build direction', 'Tangent direction', 'Normal direction'])
            else:
                labels.extend(['', '', ''])

        # Convert to numpy arrays for efficient handling
        plot_points = np.array(plot_points)
        plot_vectors = np.array(plot_vectors)

        # Add tool vectors if different from build vectors
        if tool_vectors is not None:
            # Normalize vectors for comparison
            norm_tool = tool_vectors / np.linalg.norm(tool_vectors, axis=1, keepdims=True)
            norm_build = b_vectors / np.linalg.norm(b_vectors, axis=1, keepdims=True)

            # Calculate difference between normalized vectors
            diff = np.linalg.norm(norm_tool - norm_build, axis=1)
            mask = diff > 1e-6  # Tolerance threshold

            if np.any(mask):
                # Add points where tool vector differs from build vector
                diff_points = points[mask]
                diff_vectors = tool_vectors[mask]

                plot_points = np.vstack([plot_points, diff_points])
                plot_vectors = np.vstack([plot_vectors, diff_vectors])
                colors.extend(['orange'] * np.sum(mask))
                labels.extend(['Tool direction' if i == 0 else '' for i in range(np.sum(mask))])

        # Single quiver call with all vectors
        self.ax.quiver(
            plot_points[:, 0], plot_points[:, 1], plot_points[:, 2],
            plot_vectors[:, 0], plot_vectors[:, 1], plot_vectors[:, 2],
            colors=colors,
            normalize=True
        )

        # Add legend manually for unique labels
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', label='Build direction'),
            Line2D([0], [0], color='blue', label='Tangent direction'),
            Line2D([0], [0], color='green', label='Normal direction')
        ]
        if tool_vectors is not None and np.any(diff > 1e-6):
            legend_elements.append(Line2D([0], [0], color='purple', label='Tool direction'))

        self.ax.legend(handles=legend_elements)

    #-------------------------------------------------------------------
    # Events methods
    #-------------------------------------------------------------------

    def on_key_press(self, event):
        """
        Handle keyboard events
        """
        if self.visible_points is None:
            self.update_visible_points_vectors()

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
        plt.show()

