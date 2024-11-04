from CollisionAvoidance import TrajectoryAcquisition
from Visualizer import TrajectoryVisualizer

# Create trajectory acquisition instance
trajectory = TrajectoryAcquisition("Trajectoire_Schwarz.csv")

# Create visualizer
visualizer = TrajectoryVisualizer(trajectory, 90,[0,60], 1, True, False, 1)

# Start visualization
visualizer.visualize()
