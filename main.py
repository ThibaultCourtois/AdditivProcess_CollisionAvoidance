import numpy as np
from TrajectoryDataManager import TrajectoryManager
from CollisionAvoidance import CollisionAvoidance
from Visualizer import AdvancedTrajectoryVisualizer
from MetricsVisualizer import MetricsVisualizer

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

INPUT_FILE = "Tore_WAAM.csv"
OUTPUT_FILE = "Tore_WAAM_Optimized.csv"

# Geometric parameters
bead_width = 3.0  # mm
bead_height = 1.95  # mm
tool_radius = 12.5  # mm
tool_length = 1000
nozzle_length = 17.0  # mm
n_bead_points = 16
n_tool_points = 16

# Visualization parameters
display_layers = [0, 360]  # Adjust according to the layers to display
revolut_angle = 90
stride = 1

# -------------------------------------------------------------------
# Trajectory processing & collision detection
# -------------------------------------------------------------------

print("\n=== Setting up trajectory and running collision detection ===")

# Create managers
trajectory_manager = TrajectoryManager(INPUT_FILE)
collision_manager = CollisionAvoidance(
    trajectory_path=INPUT_FILE,
    bead_width=bead_width,
    bead_height=bead_height,
    tool_radius=tool_radius,
    tool_length=nozzle_length,
)

# Génération des points candidats pour toutes les couches
print("Generating collision candidates...")
for layer_idx in range(len(trajectory_manager.layer_indices)):
    start_idx = trajectory_manager.layer_indices[layer_idx]
    end_idx = (trajectory_manager.layer_indices[layer_idx + 1]
               if layer_idx + 1 < len(trajectory_manager.layer_indices)
               else len(trajectory_manager.points))

    layer_points = trajectory_manager.points[start_idx:end_idx]
    layer_normals = trajectory_manager.n_vectors[start_idx:end_idx]
    layer_builds = trajectory_manager.b_vectors[start_idx:end_idx]

    collision_manager.collision_candidates_generator.generate_collision_candidates_per_layer(
        points=layer_points,
        normal_vectors=layer_normals,
        build_vectors=layer_builds,
        layer_index=layer_idx,
        is_last_layer  = layer_idx == len(trajectory_manager.layer_indices) - 1,
    )

print("Detecting initial collisions...")
collision_points = collision_manager.detect_collisions_optimized()
#collision_points = collision_manager.detect_collisions_exhaustive()


# -------------------------------------------------------------------
# Initial trajectory visualization
# -------------------------------------------------------------------

print("\n=== Visualizing Initial Trajectory ===")

# Create and configure visualizer for initial trajectory
initial_visualizer = AdvancedTrajectoryVisualizer(
    trajectory_manager=trajectory_manager,
    collision_manager=collision_manager,
    display_layers=display_layers,
    revolut_angle=revolut_angle,
    stride=stride
)

# Configure geometry parameters
initial_visualizer.geometry_visualizer.set_parameters(
    bead_width=bead_width,
    bead_height=bead_height,
    tool_radius=tool_radius,
    tool_length=tool_length,
    nozzle_length=nozzle_length,
    n_bead_points=n_bead_points,
    n_tool_points=n_tool_points
)

# Setup visualization options
initial_visualizer.setup_visualization(
    show_beads=False,
    low_res_bead=True,
    show_vectors=False,
    show_tool=False,
    show_collisions=True,
    show_collision_candidates=False,
    show_collision_bases=True
)

initial_visualizer.create_figure()
initial_visualizer.apply_layer_filter(
    layers=display_layers,
    angle_limit=revolut_angle
)
initial_visualizer.visualize_trajectory()
initial_visualizer.show()

# -------------------------------------------------------------------
# Trajectory optimization
# -------------------------------------------------------------------

print("\n=== Processing Trajectory Optimization ===")
new_tool_vectors = collision_manager.process_trajectory()
trajectory_manager.save_modified_trajectory(new_tool_vectors, OUTPUT_FILE)

# -------------------------------------------------------------------
# Metrics Visualization
# -------------------------------------------------------------------

print("\n=== Visualizing Metrics ===")

# Création du visualiseur de métriques
metrics_visualizer = MetricsVisualizer(collision_manager, trajectory_manager)

# Visualisation des angles de correction par couche
print("Generating tilt angles by layer visualization...")
metrics_visualizer.visualize_tilt_angles_by_layer(threshold=5)

# Visualisation des angles avec seuil
print("Generating tilt angles with threshold visualization...")
metrics_visualizer.visualize_tilt_angle_with_threshold(collision_manager, threshold=5)

# -------------------------------------------------------------------
# Optimized trajectory visualization
# -------------------------------------------------------------------

print("\n=== Visualizing Optimized Trajectory ===")

# Create new managers for optimized trajectory
optimized_trajectory = TrajectoryManager(OUTPUT_FILE)
optimized_collision_manager = CollisionAvoidance(
    trajectory_path=OUTPUT_FILE,
    bead_width=bead_width,
    bead_height=bead_height,
    tool_radius=tool_radius,
    tool_length=nozzle_length
)

# Génération des points candidats pour la trajectoire optimisée
print("Generating collision candidates for optimized trajectory...")
for layer_idx in range(len(optimized_trajectory.layer_indices)):
    start_idx = optimized_trajectory.layer_indices[layer_idx]
    end_idx = (optimized_trajectory.layer_indices[layer_idx + 1]
               if layer_idx + 1 < len(optimized_trajectory.layer_indices)
               else len(optimized_trajectory.points))

    layer_points = optimized_trajectory.points[start_idx:end_idx]
    layer_normals = optimized_trajectory.n_vectors[start_idx:end_idx]
    layer_builds = optimized_trajectory.b_vectors[start_idx:end_idx]

    optimized_collision_manager.collision_candidates_generator.generate_collision_candidates_per_layer(
        points=layer_points,
        normal_vectors=layer_normals,
        build_vectors=layer_builds,
        layer_index=layer_idx,
        is_last_layer=layer_idx == len(trajectory_manager.layer_indices) - 1
    )

optimized_collision_points = optimized_collision_manager.detect_collisions_optimized()

# Create and configure visualizer for optimized trajectory
optimized_visualizer = AdvancedTrajectoryVisualizer(
    trajectory_manager=optimized_trajectory,
    collision_manager=optimized_collision_manager,
    display_layers=display_layers,
    revolut_angle=revolut_angle,
    stride=stride
)

# Configure geometry parameters
optimized_visualizer.geometry_visualizer.set_parameters(
    bead_width=bead_width,
    bead_height=bead_height,
    tool_radius=tool_radius,
    tool_length=nozzle_length,
    n_bead_points=n_bead_points,
    n_tool_points=n_tool_points
)

# Setup visualization options
optimized_visualizer.setup_visualization(
    show_beads=False,
    low_res_bead=True,
    show_vectors=False,
    show_tool=False,
    show_collisions=True,
    show_collision_candidates=False,
    show_collision_bases=True
)

optimized_visualizer.create_figure()
optimized_visualizer.apply_layer_filter(
    layers=display_layers,
    angle_limit=revolut_angle
)
optimized_visualizer.visualize_trajectory()
optimized_visualizer.show()