import pyvista as pv

# mesh = pv.Box(level= 2)

# clipped = pv.Box([0, 1, 0, 1, 0, 1])
# clipped = clipped.clip_box([0, 0.5, 0, 0.5, 0, 0.5])


# plotter = pv.Plotter()
# plotter.add_mesh(mesh, cmap="coolwarm", opacity= 0.15, show_edges= True)
# plotter.add_mesh(clipped, color="tan", opacity= 0.5, show_edges= True)
# plotter.show()

cube = pv.Cube().triangulate().subdivide(2)
plotter = pv.Plotter()
clipped = pv.Box([0, 0.2, 0, 0.2, 0, 0.2])
plotter.add_mesh(cube, show_edges= True, opacity= 0.1)
plotter.add_mesh(clipped, show_edges= True, opacity = 0.2)
plotter.background_color = "black"
plotter.screenshot("cube.png")