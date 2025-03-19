import numpy as np
from threed.utils import rodrigues, dict_to_sparse_matrix

def compute_joint_locations(vertices, J_regressor):
    """
    Compute joint locations based on vertices and joint regressor.
    
    Formula from SMPL paper (2015):
    J = RV
    
    Args:
        vertices: Mesh vertices
        J_regressor: Joint regressor matrix
        
    Returns:
        Joint locations (24, 3)
    """
    if isinstance(J_regressor, dict):
        J_regressor = dict_to_sparse_matrix(J_regressor)
    joint_locations = J_regressor.dot(vertices)
    return joint_locations

def add_pose_deformation(shaped_vertices, posedirs, poses):
    """
    Add pose-dependent deformations to the shaped mesh.
    
    Formula from SMPL paper (2015):
    T̄p(β, θ) = T̄(β) + Bp(θ)
    
    Args:
        shaped_vertices: Vertices after applying shape blend shapes
        posedirs: Pose-dependent blend shapes
        poses: Pose parameters as rotation vectors (24, 3)
        
    Returns:
        Mesh vertices after applying pose-dependent deformations
    """
    posed_vertices = shaped_vertices.copy()
    
    if poses.shape[-1] == 3:
        rotation_matrices = np.array([rodrigues(pose) for pose in poses])
    else:
        poses_reshaped = poses.reshape((-1, 3))
        rotation_matrices = np.array([rodrigues(pose) for pose in poses_reshaped])
    
    I = np.eye(3)
    R_posed = np.zeros(24 * 9)
    
    for i in range(len(rotation_matrices)):
        R_posed[i*9:(i+1)*9] = (rotation_matrices[i] - I).flatten()
    
    R_posed = R_posed[:posedirs.shape[2]]
    
    for i in range(len(R_posed)):
        posed_vertices += posedirs[:, :, i] * R_posed[i]
    
    return posed_vertices, rotation_matrices