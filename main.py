from Animation import TrajectoryAnimation
from CollisionAvoidance import TrajectoryAcquisition
from Visualizer import TrajectoryVisualizer
import numpy as np

# Create trajectory acquisition instance
trajectory = TrajectoryAcquisition("Trajectoire_Schwarz.csv")

# Create visualizer
visualizer = TrajectoryVisualizer(trajectory, 360,[0,0], 1, True, True, 1)

# Start visualization
visualizer.visualize()
