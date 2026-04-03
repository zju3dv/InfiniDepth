from training.utils.logger import Log
import copy
import open3d as o3d
import numpy as np


def post_process_mesh(mesh, cluster_to_keep=50):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    Log.info("post processing the mesh to have {} clusterscluster_to_kep".format(
        cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_0.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    try:
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    except:
        n_cluster = np.sort(cluster_n_triangles.copy())[0]
    n_cluster = max(n_cluster, 500)  # filter meshes smaller than 50
    Log.info(f'Keep {n_cluster} clusters.')
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    Log.info("num vertices raw {}".format(len(mesh.vertices)))
    Log.info("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0
