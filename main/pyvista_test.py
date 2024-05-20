
import pyvista as pv
import numpy as np


points_Eg= {"x1":[0,0,0],
            "x2":[0,0.2,0],
            "x3":[-0.025,0.2,0],
            "x4":[-0.025,1,0],
            "x5":[0.075,1,0],
            "x6":[0.075,1.25,0],
            "x7":[1.025,1.25,0],
            "x8":[1.025,0.25,0],
            "x9":[1,0.25,0],
            "x10":[1,0,0], 
            "x1out":[0,0,0.1],
            "x2out":[0,0.2,0.1],
            "x3out":[-0.025,0.2,0.1],
            "x4out":[-0.025,1,0.1],
            "x5out":[0.075,1,0.1],
            "x6out":[0.075,1.25,0.1],
            "x7out":[1.025,1.25,0.1],
            "x8out":[1.025,0.25,0.1],
            "x9out":[1,0.25,0.1],
            "x10out":[1,0,0.1]}
Connecting_Eg= {"x1_x2":[points_Eg["x1"],points_Eg["x2"]],
                "x2_x3":[points_Eg["x2"],points_Eg["x3"]],
                "x3_x4":[points_Eg["x3"],points_Eg["x4"]],
                "x4_x5":[points_Eg["x4"],points_Eg["x5"]],
                "x5_x6":[points_Eg["x5"],points_Eg["x6"]],
                "x6_x7":[points_Eg["x6"],points_Eg["x7"]],
                "x7_x8":[points_Eg["x7"],points_Eg["x8"]],
                "x8_x9":[points_Eg["x8"],points_Eg["x9"]],
                "x9_x10":[points_Eg["x9"],points_Eg["x10"]],
                "x10_x1":[points_Eg["x10"],points_Eg["x1"]],
                "x1_x1out":[points_Eg["x1"],points_Eg["x1out"]],
                "x2_x2out":[points_Eg["x2"],points_Eg["x2out"]],
                "x3_x3out":[points_Eg["x3"],points_Eg["x3out"]],
                "x4_x4out":[points_Eg["x4"],points_Eg["x4out"]],
                "x5_x5out":[points_Eg["x5"],points_Eg["x5out"]],
                "x6_x6out":[points_Eg["x6"],points_Eg["x6out"]],
                "x7_x7out":[points_Eg["x7"],points_Eg["x7out"]],
                "x8_x8out":[points_Eg["x8"],points_Eg["x8out"]],
                "x9_x9out":[points_Eg["x9"],points_Eg["x9out"]],
                "x10_x10out":[points_Eg["x10"],points_Eg["x10out"]],
                "x1out_x2out":[points_Eg["x1out"],points_Eg["x2out"]],
                "x2out_x3out":[points_Eg["x2out"],points_Eg["x3out"]],
                "x3out_x4out":[points_Eg["x3out"],points_Eg["x4out"]],
                "x4out_x5out":[points_Eg["x4out"],points_Eg["x5out"]],
                "x5out_x6out":[points_Eg["x5out"],points_Eg["x6out"]],
                "x6out_x7out":[points_Eg["x6out"],points_Eg["x7out"]],
                "x7out_x8out":[points_Eg["x7out"],points_Eg["x8out"]],
                "x8out_x9out":[points_Eg["x8out"],points_Eg["x9out"]],
                "x9out_x10out":[points_Eg["x9out"],points_Eg["x10out"]],
                "x10out_x1out":[points_Eg["x10out"],points_Eg["x1out"]]}

points_Og= {"x1":[-0.1,-0.17,0.1],
            "x2":[-0.1,0.23,0.1],
            "x3":[-0.05,0.23,0.1],
            "x4":[-0.05,0.68,0.1],
            "x5":[-0.2,0.68,0.1],
            "x6":[-0.2,1.28,0.1],
            "x7":[0.4,1.28,0.1],
            "x8":[0.4,1.13,0.1],
            "x9":[1.1,1.13,0.1],
            "x10":[1.1,0.03,0.1],
            "x11":[1.05,0.03,0.1],
            "x12":[1.05,-0.05,0.1],
            "x13":[0.4,-0.05,0.1],
            "x14":[0.4,-0.17,0.1],
            "x1out":[-0.1,-0.17,0.2],
            "x2out":[-0.1,0.23,0.2],
            "x3out":[-0.05,0.23,0.2],
            "x4out":[-0.05,0.68,0.2],
            "x5out":[-0.2,0.68,0.2],
            "x6out":[-0.2,1.28,0.2],
            "x7out":[0.4,1.28,0.2],
            "x8out":[0.4,1.13,0.2],
            "x9out":[1.1,1.13,0.2],
            "x10out":[1.1,0.03,0.2],
            "x11out":[1.05,0.03,0.2],
            "x12out":[1.05,-0.05,0.2],
            "x13out":[0.4,-0.05,0.2],
            "x14out":[0.4,-0.17,0.2]}
Connecting_Og= {"x1_x2":[points_Og["x1"],points_Og["x2"]],
                "x2_x3":[points_Og["x2"],points_Og["x3"]],
                "x3_x4":[points_Og["x3"],points_Og["x4"]],
                "x4_x5":[points_Og["x4"],points_Og["x5"]],
                "x5_x6":[points_Og["x5"],points_Og["x6"]],
                "x6_x7":[points_Og["x6"],points_Og["x7"]],
                "x7_x8":[points_Og["x7"],points_Og["x8"]],
                "x8_x9":[points_Og["x8"],points_Og["x9"]],
                "x9_x10":[points_Og["x9"],points_Og["x10"]],
                "x10_x11":[points_Og["x10"],points_Og["x11"]],
                "x11_x12":[points_Og["x11"],points_Og["x12"]],
                "x12_x13":[points_Og["x12"],points_Og["x13"]],
                "x13_x14":[points_Og["x13"],points_Og["x14"]],
                "x14_x1":[points_Og["x14"],points_Og["x1"]],
                "x1_x1out":[points_Og["x1"],points_Og["x1out"]],
                "x2_x2out":[points_Og["x2"],points_Og["x2out"]],
                "x3_x3out":[points_Og["x3"],points_Og["x3out"]],
                "x4_x4out":[points_Og["x4"],points_Og["x4out"]],
                "x5_x5out":[points_Og["x5"],points_Og["x5out"]],
                "x6_x6out":[points_Og["x6"],points_Og["x6out"]],
                "x7_x7out":[points_Og["x7"],points_Og["x7out"]],
                "x8_x8out":[points_Og["x8"],points_Og["x8out"]],
                "x9_x9out":[points_Og["x9"],points_Og["x9out"]],
                "x10_x10out":[points_Og["x10"],points_Og["x10out"]],
                "x11_x11out":[points_Og["x11"],points_Og["x11out"]],
                "x12_x12out":[points_Og["x12"],points_Og["x12out"]],
                "x13_x13out":[points_Og["x13"],points_Og["x13out"]],
                "x14_x14out":[points_Og["x14"],points_Og["x14out"]],
                "x1out_x2out":[points_Og["x1out"],points_Og["x2out"]],
                "x2out_x3out":[points_Og["x2out"],points_Og["x3out"]],
                "x3out_x4out":[points_Og["x3out"],points_Og["x4out"]],
                "x4out_x5out":[points_Og["x4out"],points_Og["x5out"]],
                "x5out_x6out":[points_Og["x5out"],points_Og["x6out"]],
                "x6out_x7out":[points_Og["x6out"],points_Og["x7out"]],
                "x7out_x8out":[points_Og["x7out"],points_Og["x8out"]],
                "x8out_x9out":[points_Og["x8out"],points_Og["x9out"]],
                "x9out_x10out":[points_Og["x9out"],points_Og["x10out"]],
                "x10out_x11out":[points_Og["x10out"],points_Og["x11out"]],
                "x11out_x12out":[points_Og["x11out"],points_Og["x12out"]],
                "x12out_x13out":[points_Og["x12out"],points_Og["x13out"]],
                "x13out_x14out":[points_Og["x13out"],points_Og["x14out"]],
                "x14out_x1out":[points_Og["x14out"],points_Og["x1out"]]}

Raume = {"hka-aqm-a017": {"x1017":[0.85,0,0],
                "x2017":[0.85,0.25,0],
                "x3017":[1,0.25,0],
                "x4017":[1,0,0],
                "x1017out":[0.85,0,0.1],
                "x2017out":[0.85,0.25,0.1],
                "x3017out":[1,0.25,0.1],
                "x4017out":[1,0,0.1]},
        "hka-aqm-a103": {"x1103":[-0.1,-0.17,0.1],
                "x2103":[-0.1,0.23,0.1],
                "x3103":[0.4,0.23,0.1],
                "x4103":[0.4,-0.17,0.1],
                "x1103out":[-0.1,-0.17,0.2],
                "x2103out":[-0.1,0.23,0.2],
                "x3103out":[0.4,0.23,0.2],
                "x4103out":[0.4,-0.17,0.2]},
        "hka-aqm-a102": {"x1102":[-0.2,0.68,0.1],
                "x2102":[-0.2,1.28,0.1],
                "x3102":[0.4,1.28,0.1],
                "x4102":[0.4,0.68,0.1],
                "x1102out":[-0.2,0.68,0.2],
                "x2102out":[-0.2,1.28,0.2],
                "x3102out":[0.4,1.28,0.2],
                "x4102out":[0.4,0.68,0.2]},
        "hka-aqm-a112": {"x1112":[0.9,1.13,0.1],
                "x2112":[1.1,1.13,0.1],
                "x3112":[1.1,0.83,0.1],
                "x4112":[0.9,0.83,0.1],
                "x1112out":[0.9,1.13,0.2],
                "x2112out":[1.1,1.13,0.2],
                "x3112out":[1.1,0.83,0.2],
                "x4112out":[0.9,0.83,0.2]},
        "hka-aqm-a106": {"x1106":[1.05,-0.05,0.1],
                "x2106":[0.7,-0.05,0.1],
                "x3106":[0.7,0.03,0.1],
                "x4106":[1.05,0.03,0.1],
                "x1106out":[1.05,-0.05,0.2],
                "x2106out":[0.7,-0.05,0.2],
                "x3106out":[0.7,0.03,0.2],
                "x4106out":[1.05,0.03,0.2]},
        "hka-aqm-a107": {"x1107":[1.1,0.03,0.1],
                "x2107":[1.0,0.03,0.1],
                "x3107":[1.0,0.23,0.1],
                "x4107":[1.1,0.23,0.1],
                "x1107out":[1.1,0.03,0.2],
                "x2107out":[1.0,0.03,0.2],
                "x3107out":[1.0,0.23,0.2],
                "x4107out":[1.1,0.23,0.2]},
        "hka-aqm-a108": {"x1108":[1.1,0.23,0.1],
                "x2108":[1.0,0.23,0.1],
                "x3108":[1.0,0.355,0.1],
                "x4108":[1.1,0.355,0.1],
                "x1108out":[1.1,0.23,0.2],
                "x2108out":[1.0,0.23,0.2],
                "x3108out":[1.0,0.355,0.2],
                "x4108out":[1.1,0.355,0.2]},
        "Raum_109": {"x1109":[1.1,0.355,0.1],
                "x2109":[1.0,0.355,0.1],
                "x3109":[1.0,0.48,0.1],
                "x4109":[1.1,0.48,0.1],
                "x1109out":[1.1,0.355,0.2],
                "x2109out":[1.0,0.355,0.2],
                "x3109out":[1.0,0.48,0.2],
                "x4109out":[1.1,0.48,0.2]},
        "Raum_110": {"x1110":[1.1,0.48,0.1],
                "x2110":[1.0,0.48,0.1],
                "x3110":[1.0,0.605,0.1],
                "x4110":[1.1,0.605,0.1],
                "x1110out":[1.1,0.48,0.2],
                "x2110out":[1.0,0.48,0.2],
                "x3110out":[1.0,0.605,0.2],
                "x4110out":[1.1,0.605,0.2]},
        "hka-aqm-a111": {"x1111":[1.1,0.605,0.1],
                    "x2111":[1.0,0.605,0.1],
                    "x3111":[1.0,0.83,0.1],
                    "x4111":[1.1,0.83,0.1],
                    "x1111out":[1.1,0.605,0.2],
                    "x2111out":[1.0,0.605,0.2],
                    "x3111out":[1.0,0.83,0.2],
                    "x4111out":[1.1,0.83,0.2]},
        "hka-aqm-a101": {"x1101":[0.4,0.68,0.1],
                "x2101":[0.4,0.23,0.1],
                "x3101":[0.9,0.23,0.1],
                "x4101":[0.9,0.68,0.1],
                "x1101out":[0.4,0.68,0.2],
                "x2101out":[0.4,0.23,0.2],
                "x3101out":[0.9,0.23,0.2],
                "x4101out":[0.9,0.68,0.2]}
                }    


# Erstellen Sie ein Array von Punkten aus Ihrem Dictionary
points = np.array([points_Eg[key] for key in points_Eg])
cells = np.array([[2, list(points_Eg.keys()).index(key.split("_")[0]), list(points_Eg.keys()).index(key.split("_")[1])] for key in Connecting_Eg])
# Erstellen Sie das PolyData-Objekt
mesh = pv.PolyData(points,lines=cells)

points_Og_mesh= np.array([points_Og[key] for key in points_Og])
cells_Og = np.array([[2, list(points_Og.keys()).index(key.split("_")[0]), list(points_Og.keys()).index(key.split("_")[1])] for key in Connecting_Og])
mesh_Og = pv.PolyData(points_Og_mesh,lines=cells_Og)

# points_017 = np.array([Raum_017[key] for key in Raum_017])
# cells_017 = np.array([[2, list(Raum_017.keys()).index(key.split("_")[0]), list(Raum_017.keys()).index(key.split("_")[1])] for key in Connecting_017])
# mesh_017 = pv.PolyData(points_017,lines=cells_017)
# points_017 = np.array([Raum_017[key] for key in Raum_017])
# xMin, xMax, yMin, yMax, zMin, zMax = np.min(points_017[:,0]), np.max(points_017[:,0]), np.min(points_017[:,1]), np.max(points_017[:,1]), np.min(points_017[:,2]), np.max(points_017[:,2])
# point_017 = [xMin,xMax,yMin,yMax,zMin,zMax]
# mesh_017 = pv.Box(point_017)



# Erstellen Sie den Plotter und fügen Sie das Mesh hinzu
# plotter = pv.Plotter()
# plotter.add_mesh(mesh, show_edges= True, line_width= 5)
# plotter.add_mesh(mesh_017, show_edges= True, color= "red",opacity= 0.5)
# plotter.add_mesh(mesh_Og, show_edges= True, line_width= 5)


# # Zeigen Sie das Ergebnis an
# plotter.show()

def plot_cube(Raum):
    points_raum = np.array([Raume[Raum][key] for key in Raume[Raum]])
    xMin, xMax, yMin, yMax, zMin, zMax = np.min(points_raum[:,0]), np.max(points_raum[:,0]), np.min(points_raum[:,1]), np.max(points_raum[:,1]), np.min(points_raum[:,2]), np.max(points_raum[:,2])
    point_raum = [xMin,xMax,yMin,yMax,zMin,zMax]
    mesh_raum = pv.Box(point_raum)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges= True, line_width= 5)
    plotter.add_mesh(mesh_raum, show_edges= True, color= "red",opacity= 0.5)
    plotter.add_mesh(mesh_Og, show_edges= True, line_width= 5)
    return plotter
    
#plot_cube("Raum_102").show()