import numpy as np
import trimesh
import os

from .template import load_smpl_model
from .shape import add_body_shape
from .pose import compute_joint_locations, add_pose_deformation
from .skinning import apply_skinning

def run_smpl_pipeline(model_path, betas=None, poses=None, output_path=None):
    """
    Run the complete SMPL pipeline from template to final posed and shaped mesh.
    
    Args:
        model_path: Path to the SMPL model JSON file
        betas: Shape parameters (default: zeros)
        poses: Pose parameters (default: zeros = T-pose)
        output_path: Optional path to save the output mesh
        
    Returns:
        Final mesh vertices
    """
    if betas is None:
        betas = np.zeros(10)
    
    if poses is None:
        poses = np.zeros(72)  # 24 joints * 3 rotation parameters
    
    model = load_smpl_model(model_path)
    
    shaped_vertices = add_body_shape(
        model["template_mesh"], 
        model["shapedirs"], 
        betas
    )
    
    joint_locations = compute_joint_locations(
        shaped_vertices, 
        model["J_regressor"]
    )
    
    posed_vertices, rotation_matrices = add_pose_deformation(
        shaped_vertices, 
        model["posedirs"], 
        poses
    )
    
    final_vertices = apply_skinning(
        posed_vertices,
        joint_locations,
        rotation_matrices,
        model["kintree_table"],
        model["skinning_weights"]
    )
    
    if output_path:
        colors = np.zeros((len(final_vertices), 4), dtype=np.uint8)
        colors[:, -1] = 255
        
        points = trimesh.PointCloud(
            vertices=final_vertices,
            colors=colors
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            points.export(f, file_type="xyz")
    
    return final_vertices