from TrajectoryDataManager import TrajectoryManager
from CollisionAvoidance import CollisionAvoidance
from Visualizer import TrajectoryVisualizer

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

INPUT_FILE = "Trajectoire_Schwarz.csv"
OUTPUT_FILE = "Trajectoire_Schwarz_Optimized.csv"

# Geometric parameters
bead_width = 3.0  # mm
bead_height = 1.95  # mm
tool_radius = 12.5  # mm
nozzle_length = 17.0  # mm

# Visualization parameters
display_layers = [40  , 60]  # Adjust according to the layers to display
revolut_angle = 360
stride = 3

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
    tool_length=nozzle_length
)

# Detect initial collisions once
collision_points = collision_manager.detect_initial_collisions()

# -------------------------------------------------------------------
# Initial trajectory visualization
# -------------------------------------------------------------------

print("\n=== Visualizing Initial Trajectory ===")

# Create visualizer for initial trajectory
initial_visualizer = TrajectoryVisualizer(
    trajectory_manager=trajectory_manager,
    collision_manager=collision_manager,
    revolut_angle_display=revolut_angle,
    display_layers=display_layers,
    stride=stride,
    ellipse_bool=False,
    vector_bool=False,
    show_collision_candidates_bool=True,
    show_collision_candidates_segments_bool=False,
    show_problematic_trajectory_points_bool=True,
    collision_points=collision_points
)

initial_visualizer.visualize()

# -------------------------------------------------------------------
# Trajectory optimization
# -------------------------------------------------------------------

print("\n=== Processing Trajectory Optimization ===")

# Process trajectory using existing collision detection
new_tool_vectors = collision_manager.process_trajectory()

# Save optimized trajectory
trajectory_manager.save_modified_trajectory(new_tool_vectors, OUTPUT_FILE)

# -------------------------------------------------------------------
# Optimized trajectory visualization
# -------------------------------------------------------------------

print("\n=== Visualizing Optimized Trajectory ===")

# Create new trajectory manager for optimized trajectory
optimized_trajectory = TrajectoryManager(OUTPUT_FILE)

# Create new collision manager and detect collisions for verification
optimized_collision_manager = CollisionAvoidance(
    trajectory_path=OUTPUT_FILE,
    bead_width=bead_width,
    bead_height=bead_height,
    tool_radius=tool_radius,
    tool_length=nozzle_length
)

optimized_collision_points = optimized_collision_manager.detect_initial_collisions()

# Create visualizer for optimized trajectory
optimized_visualizer = TrajectoryVisualizer(
    trajectory_manager=optimized_trajectory,
    collision_manager=optimized_collision_manager,
    revolut_angle_display=revolut_angle,
    display_layers=display_layers,
    stride=stride,
    ellipse_bool=False,
    vector_bool=False,
    show_collision_candidates_bool=False,
    show_collision_candidates_segments_bool=False,
    show_problematic_trajectory_points_bool=True,
    collision_points=optimized_collision_points
)

optimized_visualizer.visualize()