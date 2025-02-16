"""
Main script for collision detection and resolution in Wire Arc Additive Manufacturing (WAAM).

This script orchestrates the complete workflow for:
1. Loading and processing trajectory data
2. Detecting potential collisions
3. Optimizing tool orientations to avoid collisions
4. Visualizing results and metrics

The process is divided into several phases:
- Initial setup and collision detection
- Visualization of initial trajectory with collisions
- Trajectory optimization
- Metrics analysis and visualization
- Final visualization of optimized trajectory
"""

import numpy as np
from TrajectoryDataManager import TrajectoryManager
from CollisionAvoidance import CollisionAvoidance
from Visualizer import AdvancedTrajectoryVisualizer
from MetricsVisualizer import MetricsVisualizer

# -------------------------------------------------------------------
# Manufacturing Parameters
# -------------------------------------------------------------------

# File paths for input trajectory and optimized output
INPUT_FILE = "Tore_WAAM.csv"
OUTPUT_FILE = "Tore_WAAM_Optimized.csv"

# Process geometric parameters
bead_width = 3.0       # Width of deposited material (mm)
bead_height = 1.95     # Height of deposited material (mm)
tool_radius = 12.5     # Radius of tool body (mm)
tool_length = 1000     # Overall length of tool (mm) (semi-infinite cylinder)
nozzle_length = 17.0   # Length of nozzle section (mm)

# Visualization resolution parameters
bead_discretization_points = 16     # Number of points for bead cross-section visualization
tool_discretization_points = 16     # Number of points for tool cylinder visualization

# Visualization range parameters
display_layers = [0, 360]  # Layer range to display
revolut_angle = 360        # Angular range for cylindrical parts
stride = 1                 # Step size for point sampling

# -------------------------------------------------------------------
# Initial Collision Detection
# -------------------------------------------------------------------

print("\n=== Setting up trajectory and running collision detection ===")

# Initialize trajectory and collision management systems
trajectory_manager = TrajectoryManager(INPUT_FILE)
collision_manager = CollisionAvoidance(
    trajectory_path=INPUT_FILE,
    bead_width=bead_width,
    bead_height=bead_height,
    tool_radius=tool_radius,
    tool_length=nozzle_length,
)

# Generate collision candidates layer by layer
print("Generating collision candidates...")
for layer_idx in range(len(trajectory_manager.layer_indices)):
    # Calculate indices for current layer
    start_idx = trajectory_manager.layer_indices[layer_idx]
    end_idx = (trajectory_manager.layer_indices[layer_idx + 1]
               if layer_idx + 1 < len(trajectory_manager.layer_indices)
               else len(trajectory_manager.points))

    # Extract layer-specific data
    layer_points = trajectory_manager.points[start_idx:end_idx]
    layer_normals = trajectory_manager.n_vectors[start_idx:end_idx]
    layer_builds = trajectory_manager.b_vectors[start_idx:end_idx]

    # Generate collision candidates for current layer
    collision_manager.collision_candidates_generator.generate_collision_candidates_per_layer(
        points=layer_points,
        normal_vectors=layer_normals,
        build_vectors=layer_builds,
        layer_index=layer_idx,
    )

# Perform initial collision detection
print("Detecting initial collisions...")
collision_points = collision_manager.detect_collisions_optimized()
#collision_points = collision_manager.detect_collisions_exhaustive()

# -------------------------------------------------------------------
# Initial Trajectory Visualization
# -------------------------------------------------------------------

print("\n=== Visualizing Initial Trajectory ===")

# Configure visualizer for initial trajectory
initial_visualizer = AdvancedTrajectoryVisualizer(
    trajectory_manager=trajectory_manager,
    collision_manager=collision_manager,
    display_layers=display_layers,
    revolut_angle=revolut_angle,
    stride=stride
)

# Set up geometry visualization parameters
initial_visualizer.geometry_visualizer.set_parameters(
    bead_width=bead_width,
    bead_height=bead_height,
    tool_radius=tool_radius,
    tool_length=tool_length,
    nozzle_length=nozzle_length,
    bead_discretization_points=bead_discretization_points,
    tool_discretization_points=tool_discretization_points
)

# Configure visualization options for initial analysis
initial_visualizer.setup_visualization(
    show_beads=False,          # Hide bead geometry for clarity
    low_res_bead=True,         # Use simplified bead representation
    show_vectors=False,        # Hide direction vectors
    show_tool=False,          # Hide tool geometry
    show_collisions=True,     # Highlight collision points
    show_collision_candidates=False,  # Hide potential collision points
    show_collision_bases=False        # Hide local coordinate systems
)

# Generate and display initial visualization
initial_visualizer.create_figure()
initial_visualizer.apply_layer_filter(
    layers=display_layers,
    angle_limit=revolut_angle
)
initial_visualizer.visualize_trajectory()
initial_visualizer.show()

# -------------------------------------------------------------------
# Trajectory Optimization
# -------------------------------------------------------------------

print("\n=== Processing Trajectory Optimization ===")

# Optimize tool orientations to avoid collisions
new_tool_vectors = collision_manager.process_trajectory()
trajectory_manager.save_modified_trajectory(new_tool_vectors, OUTPUT_FILE)

# -------------------------------------------------------------------
# Metrics Visualization
# -------------------------------------------------------------------

print("\n=== Visualizing Metrics ===")

# Initialize metrics visualization system
metrics_visualizer = MetricsVisualizer(collision_manager, trajectory_manager)

# Generate and display metrics visualizations
print("Generating tilt angles by layer visualization...")
metrics_visualizer.visualize_tilt_angles_by_layer(threshold=5)

print("Generating tilt angles with threshold visualization...")
metrics_visualizer.visualize_tilt_angle_with_threshold(collision_manager, threshold=5)

# -------------------------------------------------------------------
# Optimized Trajectory Visualization
# -------------------------------------------------------------------

print("\n=== Visualizing Optimized Trajectory ===")

# Initialize managers for optimized trajectory
optimized_trajectory = TrajectoryManager(OUTPUT_FILE)
optimized_collision_manager = CollisionAvoidance(
    trajectory_path=OUTPUT_FILE,
    bead_width=bead_width,
    bead_height=bead_height,
    tool_radius=tool_radius,
    tool_length=nozzle_length
)

# Generate collision candidates for optimized trajectory
print("Generating collision candidates for optimized trajectory...")
for layer_idx in range(len(optimized_trajectory.layer_indices)):
    # Calculate indices for current layer
    start_idx = optimized_trajectory.layer_indices[layer_idx]
    end_idx = (optimized_trajectory.layer_indices[layer_idx + 1]
               if layer_idx + 1 < len(optimized_trajectory.layer_indices)
               else len(optimized_trajectory.points))

    # Extract layer-specific data
    layer_points = optimized_trajectory.points[start_idx:end_idx]
    layer_normals = optimized_trajectory.n_vectors[start_idx:end_idx]
    layer_builds = optimized_trajectory.b_vectors[start_idx:end_idx]

    # Generate collision candidates for current layer
    optimized_collision_manager.collision_candidates_generator.generate_collision_candidates_per_layer(
        points=layer_points,
        normal_vectors=layer_normals,
        build_vectors=layer_builds,
        layer_index=layer_idx,
    )

# Verify collision resolution
optimized_collision_points = optimized_collision_manager.detect_collisions_optimized()

# Configure visualizer for optimized trajectory
optimized_visualizer = AdvancedTrajectoryVisualizer(
    trajectory_manager=optimized_trajectory,
    collision_manager=optimized_collision_manager,
    display_layers=display_layers,
    revolut_angle=revolut_angle,
    stride=stride
)

# Set up geometry visualization parameters
optimized_visualizer.geometry_visualizer.set_parameters(
    bead_width=bead_width,
    bead_height=bead_height,
    tool_radius=tool_radius,
    tool_length=nozzle_length,
    bead_discretization_points=bead_discretization_points,
    tool_discretization_points=tool_discretization_points
)

# Configure visualization options for final analysis
optimized_visualizer.setup_visualization(
    show_beads=False,           # Hide bead geometry for clarity
    low_res_bead=True,          # Use simplified bead representation
    show_vectors=False,         # Hide direction vectors
    show_tool=False,           # Hide tool geometry
    show_collisions=True,      # Highlight any remaining collisions
    show_collision_candidates=False,   # Hide potential collision points
    show_collision_bases=True          # Show local coordinate systems
)

# Generate and display optimized visualization
optimized_visualizer.create_figure()
optimized_visualizer.apply_layer_filter(
    layers=display_layers,
    angle_limit=revolut_angle
)
optimized_visualizer.visualize_trajectory()
optimized_visualizer.show()