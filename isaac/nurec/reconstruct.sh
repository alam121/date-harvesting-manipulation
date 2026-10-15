#!/usr/bin/env bash
# Stereo rosbag -> COLMAP poses (metric) -> 3DGUT training -> NuRec USDZ for Isaac Sim 5.x.
#
#   isaac/nurec/reconstruct.sh BAG SCENE_DIR [--from extract|colmap|metric|train] [3dgrut overrides...]
#
# Output: SCENE_DIR/scene_nurec.usdz, framed at the first left camera (x fwd, y left, z up, metres).
# Environment: COLMAP_BIN, GRUT_DIR, FPS (pairs/s, default 2), WIDTH (resize, default 1280),
#              MASK_BOTTOM (fraction of image height to ignore, e.g. 0.2 for the wrist camera's gripper),
#              MASK_AUTO=1 (detect the gripper instead; MASK_BOTTOM is the fallback),
#              MIN_CHANGE (drop near-duplicate frames, default 3.0; lower keeps more views).
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
COLMAP_BIN="${COLMAP_BIN:-$HOME/miniconda3/envs/colmap/bin/colmap}"
GRUT_DIR="${GRUT_DIR:-$HOME/3dgrut}"
FPS="${FPS:-2}"
WIDTH="${WIDTH:-1280}"
MASK_BOTTOM="${MASK_BOTTOM:-0}"
MASK_AUTO="${MASK_AUTO:-0}"
MIN_CHANGE="${MIN_CHANGE:-3.0}"

BAG="$(realpath "$1")"
SCENE="$(realpath -m "$2")"
shift 2
FROM=extract
if [[ "$1" == "--from" ]]; then
    FROM="$2"
    shift 2
fi
stages=(extract colmap metric train)
run_from=0
for i in "${!stages[@]}"; do [[ "${stages[$i]}" == "$FROM" ]] && run_from=$i; done
want() { for i in "${!stages[@]}"; do [[ "${stages[$i]}" == "$1" ]] && (( i >= run_from )) && return 0; done; return 1; }

# Conda activations leak into child processes (see CALVIN policy-server crash); keep tools isolated.
unset CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH

if want extract; then
    echo "==> [1/4] Extracting synced stereo pairs"
    rm -rf "$SCENE/images" "$SCENE/masks"
    (source "$HERE/jetson_ros_env.sh" && "$NUREC_ROS_PYTHON" "$HERE/extract_frames.py" "$BAG" "$SCENE" \
        --fps "$FPS" --width "$WIDTH" --mask_bottom "$MASK_BOTTOM" --min_change "$MIN_CHANGE" \
        $([[ "$MASK_AUTO" == 1 ]] && echo --mask_auto))
fi

if want colmap; then
    echo "==> [2/4] COLMAP structure-from-motion"
    rm -rf "$SCENE/database.db" "$SCENE/sparse" "$SCENE/sparse_raw"
    find "$SCENE/images" -name '*_mask.png' -delete  # 3dgrut copies from a previous train stage
    mkdir -p "$SCENE/sparse_raw"
    read -r MODEL PARAMS < <(python3 - "$SCENE/stereo_calibration.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
l = c["left"]
if c["rectified"]:
    print("PINHOLE", ",".join(f"{l[k]:.6f}" for k in ("fx", "fy", "cx", "cy")))
else:
    print("OPENCV", "")
EOF
)
    extract_args=(--ImageReader.camera_model "$MODEL" --ImageReader.single_camera_per_folder 1)
    [[ -n "$PARAMS" ]] && extract_args+=(--ImageReader.camera_params "$PARAMS")
    [[ -d "$SCENE/masks" ]] && extract_args+=(--ImageReader.mask_path "$SCENE/masks")
    "$COLMAP_BIN" feature_extractor --database_path "$SCENE/database.db" --image_path "$SCENE/images" \
        "${extract_args[@]}" --SiftExtraction.use_gpu 1
    # Exhaustive matching also finds wide-baseline and loop-closure pairs, which SfM needs to
    # initialise when frames are dense; beyond ~600 images use the temporal stereo pair list.
    if (( $(find "$SCENE/images" -name '*.jpg' | wc -l) <= 600 )); then
        "$COLMAP_BIN" exhaustive_matcher --database_path "$SCENE/database.db" --SiftMatching.use_gpu 1
    else
        "$COLMAP_BIN" matches_importer --database_path "$SCENE/database.db" \
            --match_list_path "$SCENE/pairs.txt" --match_type pairs --SiftMatching.use_gpu 1
    fi
    refine=1
    [[ "$MODEL" == "PINHOLE" ]] && refine=0  # rectified CameraInfo intrinsics are already calibrated
    "$COLMAP_BIN" mapper --database_path "$SCENE/database.db" --image_path "$SCENE/images" \
        --output_path "$SCENE/sparse_raw" \
        --Mapper.ba_refine_focal_length "$refine" --Mapper.ba_refine_extra_params "$refine" \
        --Mapper.ba_global_frames_ratio 1.4 --Mapper.ba_global_points_ratio 1.4
    # (Global BA after 40% model growth instead of 10%: exhaustive matching on ~350 images
    # otherwise spends ~1 h in the mapper. GPU BA is unavailable: this Ceres lacks cuDSS.)

    best="" best_count=0
    for model in "$SCENE"/sparse_raw/*/; do
        count="$("$COLMAP_BIN" model_analyzer --path "$model" 2>&1 | sed -n 's/.*Registered images: *\([0-9]*\).*/\1/p')"
        echo "    model $(basename "$model"): ${count:-0} images"
        if (( ${count:-0} > best_count )); then best="$model" best_count="$count"; fi
    done
    total="$(find "$SCENE/images" -name '*.jpg' | wc -l)"
    [[ -z "$best" ]] && { echo "[ERROR] COLMAP registered no images." >&2; exit 1; }
    echo "    using $best ($best_count / $total images registered)"
    mkdir -p "$SCENE/sparse_colmap"
    "$COLMAP_BIN" model_converter --input_path "$best" --output_path "$SCENE/sparse_colmap" --output_type TXT
fi

if want metric; then
    echo "==> [3/4] Metric scale from stereo baseline, first-left-camera frame"
    rm -rf "$SCENE/sparse" "$SCENE/sparse_metric_txt"
    python3 "$HERE/metric_frame.py" "$SCENE/sparse_colmap" "$SCENE/sparse_metric_txt" "$SCENE/stereo_calibration.json"
    mkdir -p "$SCENE/sparse/0"
    "$COLMAP_BIN" model_converter --input_path "$SCENE/sparse_metric_txt" --output_path "$SCENE/sparse/0" --output_type BIN
fi

if want train; then
    echo "==> [4/4] 3DGUT training + NuRec export"
    if [[ -d "$SCENE/masks" ]]; then
        # 3dgrut reads <image>_mask.png beside each image and ignores zero pixels in the loss.
        for mask in "$SCENE"/masks/*/*.jpg.png; do
            rel="${mask#"$SCENE"/masks/}"
            cp "$mask" "$SCENE/images/${rel%.jpg.png}_mask.png"
        done
    fi
    (
        cd "$GRUT_DIR"
        source .venv/bin/activate
        python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
            path="$SCENE" out_dir="$SCENE/runs" experiment_name=nurec \
            export_usd.enabled=true export_usd.format=nurec \
            export_usd.apply_normalizing_transform=false \
            export_usd.path="$SCENE/scene_nurec.usdz" \
            "$@"
        # With PPISP, the trainer's NuRec export ignores export_usd.ppisp-integration-mode and
        # writes runtime SPG shaders Isaac 5.1 cannot use; re-export with PPISP baked into SH.
        if [[ " $* " == *" post_processing.method=ppisp "* ]]; then
            checkpoint="$(ls -dt "$SCENE"/runs/nurec/*/ | head -1)ckpt_last.pt"
            python -m threedgrut.export.scripts.export_usd -c "$checkpoint" -o "$SCENE/scene_nurec.usdz" \
                --format nurec --no-transform --ppisp-integration-mode sh-optimized --dataset "$SCENE"
        fi
    )
    echo "[DONE] $SCENE/scene_nurec.usdz"
    echo "       python isaac/isaac_hardware_scene.py --enable_cameras --nurec_usdz $SCENE/scene_nurec.usdz --nurec_pose X Y Z QW QX QY QZ"
fi
