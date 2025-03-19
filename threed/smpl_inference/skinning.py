import numpy as np
from threed.utils import create_transformation_matrix

def compute_transformation_matrices(joint_locations, rotation_matrices, kintree_table):
    """
    Compute the transformation matrices for each joint based on the kinematic tree.
    
    Formula from SMPL paper (2015):
    G_n(θ, J) = Π_k∈A(n) G_k(θ, J)
    
    Args:
        joint_locations: Joint locations (24, 3)
        rotation_matrices: Rotation matrices for each joint (24, 3, 3)
        kintree_table: Kinematic tree structure
        
    Returns:
        Transformation matrices for each joint
    """
    G = [create_transformation_matrix(rotation_matrices[0], joint_locations[0].flatten())]
    
    for i in range(1, len(joint_locations)):
        parent = kintree_table[0, i]
        
        joint_t = joint_locations[i].flatten()
        parent_joint_t = joint_locations[parent].flatten()
        
        rel_t = joint_t - parent_joint_t
        
        rel_transform = create_transformation_matrix(rotation_matrices[i], rel_t)
        
        G.append(G[parent] @ rel_transform)
    
    return np.stack(G)

def apply_skinning(posed_vertices, joint_locations, rotation_matrices, kintree_table, skinning_weights):
    """
    Apply linear blend skinning to get final vertex positions.
    
    Formula from SMPL paper (2015):
    T_p(β, θ) = ∑_k w_k,i G'_k(θ, J(β))T̄_p(β, θ)
    where G'_k = G_k(θ, J)G_k(0, J)^-1
    
    Args:
        posed_vertices: Vertices after pose-dependent deformations
        joint_locations: Joint locations
        rotation_matrices: Rotation matrices for each joint
        kintree_table: Kinematic tree structure
        skinning_weights: Skinning weights
        
    Returns:
        Final vertices after skinning
    """
    G = compute_transformation_matrices(joint_locations, rotation_matrices, kintree_table)
    
    G_inverse = np.stack([
        create_transformation_matrix(np.eye(3), -joint_locations[i].flatten()) 
        for i in range(len(joint_locations))
    ])
    
    T = np.stack([
        G[i].dot(G_inverse[i])
        for i in range(len(joint_locations))
    ])
    
    v_final = np.zeros_like(posed_vertices)
    
    for i in range(len(joint_locations)):
        w = skinning_weights[:, i].reshape(-1, 1)
        
        transformed = (T[i, :3, :3] @ posed_vertices.T).T + T[i, :3, 3]
        
        v_final += w * transformed
    
    return v_final