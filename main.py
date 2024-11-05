from CollisionAvoidance import TrajectoryAcquisition
from Visualizer import TrajectoryVisualizer

# Create trajectory acquisition instance
trajectory = TrajectoryAcquisition("Trajectoire_Schwarz.csv")

# Create visualizer
visualizer = TrajectoryVisualizer(trajectory, 180,[10,11], 1, False, True, 1)

# Tool visualization
visualizer.set_tool_visualization(show_tool=True, tool_type='cylinder')
visualizer.set_tool_parameters(radius=5, height=50, nozzle_length=3)

# Start visualization
visualizer.visualize()
