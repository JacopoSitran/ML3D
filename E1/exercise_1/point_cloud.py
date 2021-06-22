"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    # ###############
    # TODO: Implement
    sampled_points = []
    surface_areas = []
    tot_triangles = faces.shape[0]
    
    
    #find the surface areas
    for i in range(0,tot_triangles):
        vert1 = np.array([vertices[(faces[i][0]*3 + 0)],vertices[(faces[i][0]*3 + 1)],vertices[(faces[i][0]*3 + 2)]])
        vert2 = np.array([vertices[(faces[i][1]*3 + 0)],vertices[(faces[i][1]*3 + 1)],vertices[(faces[i][1]*3 + 2)]])
        vert3 = np.array([vertices[(faces[i][2]*3 + 0)],vertices[(faces[i][2]*3 + 1)],vertices[(faces[i][2]*3 + 2)]])

        
        side1 = (np.sum((vert1-vert2)**2))**0.5
        side2 = (np.sum((vert2-vert3)**2))**0.5
        side3 = (np.sum((vert3-vert1)**2))**0.5
        
        semi_peri = (side1+side2+side3)/2
        area_tri = (semi_peri*(semi_peri-side1)*(semi_peri-side2)*(semi_peri-side3))**0.5
        surface_areas.append(area_tri)
        

    surface_area_probs = surface_areas/sum(surface_areas)
    triangle_iter = np.random.choice(tot_triangles,n_points,p=surface_area_probs)
    
    for i in triangle_iter:
        vert1 = np.array([vertices[(faces[i][0]*3 + 0)],vertices[(faces[i][0]*3 + 1)],vertices[(faces[i][0]*3 + 2)]])
        vert2 = np.array([vertices[(faces[i][1]*3 + 0)],vertices[(faces[i][1]*3 + 1)],vertices[(faces[i][1]*3 + 2)]])
        vert3 = np.array([vertices[(faces[i][2]*3 + 0)],vertices[(faces[i][2]*3 + 1)],vertices[(faces[i][2]*3 + 2)]])

        
        r1 = np.random.uniform(0,1,1)
        r2 = np.random.uniform(0,1,1)
        u = 1- (r1**0.5)
        v = (1-r2)*(r1**0.5)
        w = (r1**0.5)*r2
        P = (u*vert1) + (v*vert2) + (w*vert3)
        sampled_points.append(P)
    
    
    # total variables
    return np.array(sampled_points)
    
    # ###############
