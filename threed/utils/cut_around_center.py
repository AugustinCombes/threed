import numpy as np
import trimesh

def cut_around_center(point_cloud, threshold=0.2):
    vertices = point_cloud.vertices[:, [1, 0, 2]] * np.array([1, -1, 1])
    distances = np.linalg.norm(vertices[:, [0, -1]] - point_cloud.centroid[[0, -1]], axis=-1)
    filtered_vertices = vertices[distances < threshold]
    filtered_colors = point_cloud.colors[distances < threshold]
    point_cloud.vertices = filtered_vertices
    point_cloud.colors = filtered_colors
    return point_cloud
    
if __name__ == "__main__":
    scene = trimesh.load_scene(
        "/Users/gus/Desktop/threed/output.glb", 
        "glb"
    )
    filtered_point_cloud = cut_around_center(scene.geometry["geometry_0"])
    with open("filtered_output.xyz", "wb") as f:
        filtered_point_cloud.export(f, file_type="xyz")
