"""Export to disk"""

def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # TODO: Implement
    # ###############
    vertices = vertices.reshape(-1,3)
    faces+=1
    file = open(path, 'w')
    for vertex in vertices:
        file.write("v {0} {1} {2}\n".format(vertex[0],vertex[1],vertex[2]))

    for face in faces:
        file.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(face[0],face[1],face[2]))  

    file.close()

def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    # TODO: Implement
    f = open(path, "w")
    for i in range(0,pointcloud.shape[0]):
        if (i!=0):
            f.write("\n")
        f.write("v")
        for j in range(0,pointcloud.shape[1]):
            f.write(" ")
            f.write(str(pointcloud[i,j]))
    f.close()
    # ###############
    
