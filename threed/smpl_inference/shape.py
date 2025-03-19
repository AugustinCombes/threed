import numpy as np

def add_body_shape(template_mesh, shapedirs, betas):
    """
    Apply shape blend shapes to the template mesh based on shape parameters (betas).
    
    Formula from SMPL paper (2015):
    T̄(β) = T̄ + Bs(β)
    
    Args:
        template_mesh: Base mesh vertices in T-pose
        shapedirs: Shape blend shapes directions
        betas: Shape parameters (typically 10-dimensional)
        
    Returns:
        Mesh vertices after applying shape blend shapes
    """
    shaped_mesh = template_mesh.copy()
    
    for i, beta in enumerate(betas):
        if i >= shapedirs.shape[2]:
            break
        shaped_mesh += beta * shapedirs[:, :, i]
    
    return shaped_mesh