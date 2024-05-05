import pyvista as pv
import numpy as np
import os
os.environ["PYVISTA_OFF_SCREEN"]="true"
os.environ["PYVISTA_USE_PANEL"]="true"
os.environ["PYVISTA_PLOT_THEME"] = "document"  # or any other valid theme
os.environ["PYVISTA_AUTO_CLOSE"]="false"

# Create cube vertices
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
])

# Create cube faces
faces = np.array([
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]
])

# Create cube polydata
cube = pv.PolyData(vertices, faces)

# Create scatter points
n_points = 100
points = np.random.rand(n_points, 3) * 0.8 + 0.1  # Random points within the cube

# Create scatter points polydata
scatter = pv.PolyData(points)

# Create plotter
plotter = pv.Plotter()

# Add cube and scatter points to the plotter
plotter.add_mesh(cube, color='blue', opacity=0.5)
plotter.add_points(scatter, color='red', point_size=5)

# Display the plotter
plotter.show(auto_close=False)