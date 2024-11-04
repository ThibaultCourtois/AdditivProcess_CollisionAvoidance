from CollisionAvoidance import TrajectoryAcquisition
from Visualizer import TrajectoryVisualizer

# Create trajectory acquisition instance
trajectory = TrajectoryAcquisition("Trajectoire_Schwarz.csv")

# Create visualizer
visualizer = TrajectoryVisualizer(trajectory, 90,[55,50], 1, False, True, 1)

# Start visualization
visualizer.visualize()
