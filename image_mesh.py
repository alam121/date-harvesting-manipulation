from pathlib import Path
import subprocess
import open3d as o3d
import os
import time

def run_colmap(command, label=None):
    command = [str(c) for c in command]  # ensure string
    if label:
        print(f"\n Starting: {label}")
    print(f"Running: {' '.join(command)}")
    start = time.time()
    subprocess.run(command, check=True)
    end = time.time()
    if label:
        print(f" Done: {label} ({end - start:.2f} sec)")

# To reduce computation time, you can lower the following values:
# - Maximum image size: max_image_size [500 ... 2000]
# - Number of iterations: PatchMatchStereo.num_iterations [2 ... 5]
# - Number of samples: PatchMatchStereo.num_samples [3 ... 15]

def run():
    total_start = time.time()
    output_path = Path("/home/moon/workspace/colmap_ws/qassim_ws_5/") # Path to a new folder for saving mesh and other output files
    image_path ="/home/moon/workspace/colmap_ws/videos/Qassim2" # Path to the folder with the images
    database_path = output_path / "database.db"
    sfm_path = output_path / "sparse"
    dense_path = output_path / "dense"
    dense_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(exist_ok=True)

    # Step 1: Feature extraction
    run_colmap([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", str(image_path)
    ], label="Feature Extraction")

    # Step 2: Feature matching
    run_colmap([
        "colmap", "exhaustive_matcher",
        "--database_path", database_path
    ], label="Feature Matching")

    # Step 3: Sparse reconstruction (mapper)
    os.makedirs(sfm_path, exist_ok=True)
    
    run_colmap([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", str(image_path),
        "--output_path", sfm_path
    ], label="Sparse Reconstruction")

    # 1. Undistort images
    run_colmap([
        "colmap", "image_undistorter",
        "--image_path", str(image_path),
        "--input_path", str(sfm_path/"0"),
        "--output_path", str(dense_path),
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ], label="Undistort")

    # 2. PatchMatch stereo
    run_colmap([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_path),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.depth_min", "0.001",
        "--PatchMatchStereo.depth_max", "3.0",
        "--PatchMatchStereo.window_step", "1",
        "--PatchMatchStereo.num_samples", "15",
        "--PatchMatchStereo.num_iterations", "5"
    ], label="PatchMatch")

    # 3. Stereo
    run_colmap([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_path),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_path / "fused.ply")
    ], label="Stereo")

    # 4. Poisson
    run_colmap([
        "colmap", "poisson_mesher",
        "--input_path", str(dense_path / "fused.ply"),
        "--output_path", str(dense_path / "meshed_poisson.ply")
    ], label="Poisson")

    total_end = time.time()
    print(f"\n Total pipeline time: {total_end - total_start:.2f} sec")

    mesh = o3d.io.read_point_cloud(str(dense_path / "fused.ply"))
    o3d.visualization.draw_geometries([mesh])

    mesh = o3d.io.read_point_cloud(str(dense_path / "meshed_poisson.ply"))
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    run()
