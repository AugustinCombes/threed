import numpy as np

def rodrigues(r):
    """
    Rodrigues formula: Convert axis-angle representation to rotation matrix.
    
    Args:
        r: Axis-angle representation (3,) array
        
    Returns:
        Rotation matrix (3, 3)
    """
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        return np.eye(3)
    
    # Normalize rotation axis
    k = r / theta
    
    # Rodrigues formula
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def create_transformation_matrix(R, t):
    """
    Create a 4x4 transformation matrix from rotation R and translation t.
    
    Args:
        R: 3x3 rotation matrix
        t: 3D translation vector
    
    Returns:
        4x4 transformation matrix
    """
    t = np.asarray(t).flatten()
    top_block = np.concatenate([R, t.reshape(3, 1)], axis=1)
    bottom_row = np.array([[0, 0, 0, 1]])
    T = np.concatenate([top_block, bottom_row], axis=0)
    
    return T