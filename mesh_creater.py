from pathlib import Path
import subprocess
import open3d as o3d
import os
import time
import pyzed.sl as sl
import cv2

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

# During recording, the robot must capture one image per second 
# (20 images over 20 seconds) from different angles around the tree

#  To reduce computation time, you can lower the following values:
# - Maximum image size: max_image_size [500 ... 2000]
# - Number of iterations: PatchMatchStereo.num_iterations [2 ... 5]
# - Number of samples: PatchMatchStereo.num_samples [3 ... 15]

def run():
    total_start = time.time()
    output_path = Path("/home/moon/workspace/1_ws/") # Path to a new folder for saving mesh and other output files
    output_path.mkdir(exist_ok=True)
    image_path = output_path / "images/"
    image_path.mkdir()

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1200  
    init_params.camera_fps = 60  

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    last_save_time = time.time()
    counter = 0
    while counter < 20:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            current_time = time.time()
            if current_time - last_save_time >= 1.0: 
                zed.retrieve_image(image, sl.VIEW.LEFT)
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                filename = image_path / f"image_{timestamp_str}.png"
                cv2.imwrite(str(filename), image.get_data())
                print(f"Saved image at {timestamp_str}")
                last_save_time = current_time
                counter+=1

    database_path = output_path / "database.db"
    sfm_path = output_path / "sparse"
    dense_path = output_path / "dense"
    dense_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Feature extraction
    run_colmap([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path
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
        "--image_path", image_path,
        "--output_path", sfm_path
    ], label="Sparse Reconstruction")

    # 1. Undistort images
    run_colmap([
        "colmap", "image_undistorter",
        "--image_path", image_path,
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
        "--PatchMatchStereo.depth_min", "0.1",
        "--PatchMatchStereo.depth_max", "2.0",
        "--PatchMatchStereo.window_step", "2",
        "--PatchMatchStereo.num_samples", "15",
        "--PatchMatchStereo.num_iterations", "5"
    ], label="PatchMatch")

    # 3. Stereo fusion
    run_colmap([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_path),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_path / "fused.ply")
    ], label="Stereo")


    total_end = time.time()
    print(f"\n Total pipeline time: {total_end - total_start:.2f} sec")

    mesh = o3d.io.read_point_cloud(str(dense_path / "fused.ply"))
    o3d.visualization.draw_geometries([mesh])
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-15, -15, 0),
        max_bound=(15, 15, 100)
    )

    cropped = mesh.crop(bbox)
    o3d.visualization.draw_geometries([cropped])

if __name__ == "__main__":
    run()
