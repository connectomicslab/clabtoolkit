import numpy as np
import nibabel as nib

def _vert_neib(faces, max_neib:int=100):
    """
    Returns a list of neighboring vertices for each vertex in a mesh
    Parameters
    ----------
    faces : numpy array of shape (n_faces, M). M is the number of vertices

    Returns
    -------
    vert_neib : numpy array of shape (n_vertices, max_neib)
        Each row contains the indices of the neighboring vertices
        for the corresponding vertex
    """
    n_vert = np.max(faces) + 1

    # Create an empty numpy array of n_vert x 100
    vert_neib = np.zeros((n_vert, max_neib), dtype=int)
    for i in range(n_vert):
        # Find all faces that contain vertex i
        faces_with_i = np.where(faces == i)[0]
        
        # Find all vertices that are connected to vertex i
        temp = np.unique(faces[faces_with_i, :])

        # Remove vertex i from the list
        temp = temp[temp != i]
        n_neib = len(temp)
        
        # add the vertex index and the number of neighbors in front of temp array
        temp = np.hstack((i, n_neib, temp))

        # add temp array to vert_neib array
        vert_neib[i, :len(temp)] = temp 

    # Remove the colums that are all zeros
    vert_neib = vert_neib[:, ~np.all(vert_neib == 0, axis=0)]

    return vert_neib

def _annot2pyvista(annot_file):

    """
    Reads a FreeSurfer annotation file and returns the vertex labels and colors
    Parameters
    ----------
    annot_file : str
        Path to the annotation file

    Returns
    -------
    vert_lab_ord : numpy array of shape (n_vertices,)
        Vertex labels ordered by the vertex index
    reg_colors : list of hex color codes
        List of colors for each region

    """

    # Read the annotation file
    vert_lab, reg_ctable, reg_names = nib.freesurfer.read_annot(annot_file)

    # Labels for all the regions
    sts           = np.unique(vert_lab)

    # Relabelling to assing the correct colors
    vert_lab_ord = np.full(len(vert_lab), 0)

    reg_colors = []

    # Loop along all the regions
    for i, st in enumerate(sts):
        print(st)
        ind = np.where(vert_lab == st)[0] # Find the indices of the vertices with label st

        vert_lab_ord[ind] = i

        if st != -1:
            col2append = "#{:02x}{:02x}{:02x}".format(reg_ctable[st, 0], reg_ctable[st, 1], reg_ctable[st, 2])
        else:
            col2append = "#{:02x}{:02x}{:02x}".format(240, 240, 240)

        reg_colors.append(col2append)

    
    # Convert the region names to utf-8
    reg_names =  [name.decode('utf-8') for name in reg_names]

    
    return vert_lab_ord, reg_colors, reg_names

