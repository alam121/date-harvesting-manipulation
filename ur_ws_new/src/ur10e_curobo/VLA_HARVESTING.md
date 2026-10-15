# VLA integration for harvesting accuracy

## Local readiness check

The inspected machine is an NVIDIA Jetson AGX Orin Developer Kit. LeRobot is not installed in the current Python environment. The default `~/harvest_logs/episodes` directory contains 19 episode JSON files: 11 labeled GRABBED and 8 labeled NO_GRAB; 13 have `crop_close.jpg`. These are automatic outcome labels, not independently verified ground truth. Additional demonstrations may exist elsewhere.

`grasp_episode_recorder.py` records outcome-time robot state and selected snapshots. It does not record the synchronized observation/action sequence required for imitation learning. A successful final snapshot cannot tell a policy which correction produced that success. These records can support an initial outcome-classifier experiment, but are not yet a VLA training dataset.

## Proposed first scope

Keep target selection, approach planning, home, reverse, and dropoff in the existing pipeline. Train a policy for the short final approach only, using close-range camera observations and robot state to propose incremental alignment actions. Keep the force-responsive gripper controller responsible for closure execution.

Initial model candidate: SmolVLA, pending confirmation of available training/inference compute. Its official recipe supports fine-tuning with camera views, robot state, and a task instruction. This is a candidate, not a claim that its pretrained checkpoint controls this UR10e or gripper correctly.

Sources: [SmolVLA documentation](https://huggingface.co/docs/lerobot/smolvla), [OpenPI alternative](https://github.com/Physical-Intelligence/openpi).

## Required training record

For each acquisition/action step, retain:

- Unannotated camera image and acquisition timestamp; preserve crop transform/intrinsics if cropping.
- Named arm joints, measured TCP pose, gripper state, and their timestamps.
- Issued action and execution timing, distinguishing desired commands from measured motion. A multi-second trajectory command must not be mistaken for a single instantaneous action.
- Target identity, episode boundary, phase, demonstration source, and language instruction.
- Human-reviewed success/failure, damage, and intervention labels when available.

Proposed action representation: small TCP translation/rotation increments in one explicitly defined frame, plus a gripper request. Fine-tune on that exact representation; do not reinterpret a checkpoint's native joint actions as TCP deltas. Existing force inputs would require explicit state encoding and training; they are not automatically supported by a pretrained policy.

Demonstrate accurate alignment and recovery across positions, orientations, lighting, and occlusion. Existing automatic behavior can supply baseline examples, but copying its mistakes does not establish improved accuracy. Keep failure examples for outcome evaluation; use demonstrated corrective actions as imitation targets.

## Integration and evaluation

1. Extend recording to synchronized sequences and inspect alignment, labels, and action conventions before training. Keep recording off the motion callback's critical path.
2. Fine-tune in a separate training environment, preferably on an available workstation/server GPU. Avoid changing the deployed Jetson's CUDA/PyTorch stack while configuring training.
3. Replay held-out recordings and log policy proposals without commanding the robot. Split by bunch/session, not neighboring frames from the same demonstration.
4. Connect timestamped proposals to the final-approach motion interface, with matching target identity, bounded cumulative displacement, and existing constraint checks. No direct motor-torque interface is proposed.
5. Evaluate on held-out physical harvests: first-attempt success, grasp-position error where measurable, fruit damage, and time per successful harvest including retries and policy latency. Compare with the corrected existing pipeline and a simple visual-servo baseline.

Speed remains a requirement. Run inference in a separate worker with a bounded latest-observation queue and reject late proposals. Benchmark coexistence with ZED depth/YOLO/cuRobo on the Orin; process separation alone does not eliminate GPU contention. Use a separate inference GPU if available and if measured network latency is acceptable.

No VLA has been installed, trained, or enabled for motion. Hardware availability and any existing full demonstrations are the next inputs needed to choose a concrete training/deployment configuration.
