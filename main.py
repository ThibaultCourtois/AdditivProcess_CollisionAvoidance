from CollisionAvoidance import TrajectoryAcquisition
from Visualizer import TrajectoryVisualizer

# Create trajectory acquisition instance
trajectory = TrajectoryAcquisition("Trajectoire_Schwarz.csv")

# Create visualizer
visualizer = TrajectoryVisualizer(trajectory, 180,[10,11], 10, False, True, 1)
visualizer.set_tool_parameters(radius=5, height=50, nozzle_length=3)

# Start visualization
visualizer.visualize()
