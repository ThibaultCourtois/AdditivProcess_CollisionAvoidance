from CollisionAvoidance import TrajectoryAcquisition, CollisionCandidatesGenerator, CollisionAvoidance, Tool
from Visualizer import TrajectoryVisualizer
import matplotlib.pyplot as plt
import numpy as np
import time

# -------------------------------------------------------------------
# For calculus
# -------------------------------------------------------------------

# Create trajectory acquisition instance
trajectory = TrajectoryAcquisition("Trajectoire_Schwarz.csv")

# Paramètres
bead_width = 3.0  # mm
bead_height = 1.95  # mm
tool_radius = 5.0  # mm
nozzle_length = 10.0  # mm

collision_manager = CollisionAvoidance(
    trajectory_data=trajectory,
    bead_width=bead_width,
    bead_height=bead_height,
    tool_radius=tool_radius,
    tool_length=nozzle_length
)

collision_manager.process_trajectory()

# # Create visualizer
# visualizer = TrajectoryVisualizer(
#     trajectory_data=trajectory,
#     revolut_angle_display=360,  # Vue complète
#     display_layers=[0, 59],  # Afficher les couches avec collision
#     stride=1000,
#     ellipse_bool=False,
#     vector_bool=False,
#     show_collision_points_bool=False,
#     show_collision_points_segments_bool=True,
#     scale_vectors=0.1
# )
#
# # Configuration de l'outil
# visualizer.set_tool_parameters(radius=5, height=50, nozzle_length=10)
# visualizer.show_tool = True  # Activer l'affichage de l'outil88646468
#
# # Start visualization
# visualizer.visualize()

