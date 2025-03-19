import json
import numpy as np

def load_smpl_model(model_path):
    """
    Load SMPL model weights from a JSON file.
    
    Args:
        model_path: Path to the SMPL model JSON file
        
    Returns:
        Dictionary containing model weights and parameters
    """
    with open(model_path) as f:
        weights = json.load(f)
    
    template_mesh = np.array(weights["v_template"])
    
    return {
        "weights": weights,
        "template_mesh": template_mesh,
        "shapedirs": np.array(weights["shapedirs"]),
        "posedirs": np.array(weights["posedirs"]),
        "J_regressor": weights["J_regressor"],
        "kintree_table": np.array(weights["kintree_table"]),
        "skinning_weights": np.array(weights["weights"])
    }