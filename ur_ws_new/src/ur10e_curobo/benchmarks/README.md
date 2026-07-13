# GPU handoff handshake — benchmark

Measures the vision→motion GPU handoff (commit *"Replace timed vision GPU
handoff with a drain-and-ack handshake"*) using an **SVO recording**, so the
exact same frames feed YOLO on every run and results are reproducible. No robot
or cuRobo required — a small harness plays the motion node's pause side.

## What it produces

- **YOLO inference time per frame** vs the old fixed 120 ms sleep — every frame
  above the line is one the old code would have let overlap cuRobo on the GPU.
- **Handshake drain-and-ack latency** — how long the new path actually blocks
  (variable, driven by real inference), vs the old constant 120 ms.

## 1. Record (or reuse) an SVO

Any ZED `.svo` of a date-bunch scene works. Use the same file for every run.

## 2. Run the vision node against the SVO, with timing enabled

```bash
UR10E_YOLO_TIMING_CSV=/tmp/yolo_timing.csv \
    ros2 run ur10e_curobo vision --svo /path/to/scene.svo
```

`UR10E_YOLO_TIMING_CSV` is the only switch — unset it and there is zero overhead
and no file written (safe to leave the instrumentation in place).

## 3. Run the handshake harness (separate terminal)

```bash
python3 benchmarks/bench_gpu_handshake.py \
    --cycles 50 --out /tmp/handshake_latency.csv
```

It prints a summary including how many cycles exceeded 120 ms (the cases the old
fixed sleep was gambling on) and writes per-cycle latency to CSV.

## 4. Plot

```bash
python3 benchmarks/plot_handshake_results.py \
    --yolo /tmp/yolo_timing.csv \
    --handshake /tmp/handshake_latency.csv \
    --outdir /tmp/handshake_plots
```

PNGs land in `/tmp/handshake_plots/`:
- `yolo_inference_scatter.png` — the headline slide image
- `yolo_inference_hist.png`
- `handshake_latency_hist.png`

## Interpreting for the presentation

- **Headline claim:** the fixed 120 ms sleep was unsafe because YOLO inference
  time is *variable* — point to every red dot above the line in the scatter.
- **The fix:** the handshake blocks exactly as long as the drain needs, then
  proceeds deterministically (latency histogram).
- **Honest caveat:** this measures the *timing* that made overlap possible.
  Directly counting eliminated crashes needs cuRobo competing for the GPU in the
  loop (full two-node run) — the timing data shows the risk the handshake removes.
