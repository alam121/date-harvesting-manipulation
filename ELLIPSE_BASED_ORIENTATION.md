# Ellipse-Based Orientation with Relative Rotation

**Date**: 2025-12-10
**Status**: ✅ Implemented

---

## Overview

The system now uses **ellipse short axis** as the base target orientation, and sends the rotation angle needed to go from the **robot's current orientation** to this **target orientation**.

---

## How It Works

### 1. Ellipse Provides Target Orientation

For each detected fruit, the vision system computes:
- **Ellipse short axis** (PCA of fruit mask) → Target gripper width direction
- Projects this axis onto the plane perpendicular to approach direction

### 2. Rotation Angle = "How much to rotate from current pose to ellipse orientation"

The vision system computes the angle between:
- **Reference direction** (arbitrary baseline in camera frame)
- **Ellipse axis projection** (target direction)

This angle represents the rotation needed in the camera frame.

### 3. Robot Applies Rotation Relative to Current Pose

The robot receives this angle and:
1. Gets its current end-effector orientation (0° baseline)
2. Multiplies quaternions: `final = current × rotation`
3. Moves to final orientation during grasp phase

---

## Two Cases

### Sparse Bunches (z_std < 0.03)

**Rotation angle** = angle to reach ellipse orientation

```
Example:
- Ellipse axis: 45° in camera frame
- Current robot orientation: doesn't matter (robot handles it)
- Vision sends: rotation_angle = 45°
- Robot rotates: current + 45° = ellipse orientation
```

**Log output**:
```bash
[ORIENT] Sparse bunch (z_std=0.0007), ellipse_rotation=+45.0°
```

### Dense Bunches (z_std ≥ 0.03)

**Rotation angle** = ellipse angle + adjustment (±45°)

Tests 7 angles around ellipse orientation: `-45°, -30°, -15°, 0°, +15°, +30°, +45°`

Selects the one with minimum neighbor collisions.

```
Example:
- Ellipse axis: 45° in camera frame
- Collision optimization: +30° adjustment needed
- Total angle: 45° + 30° = 75°
- Vision sends: rotation_angle = 75°
- Robot rotates: current + 75° = optimized orientation
```

**Log output**:
```bash
[ORIENT] Dense bunch (z_std=0.048), ellipse_rotation=+45.0° + adjustment=+30° = +75.0° (score=8.45)
```

---

## Key Points

### What the Vision System Sends

**Topic**: `/external_goal_pose`

**Orientation quaternion** ([date_v1.6.py:962-971](ur_ws_new/src/zed_date_detector/date_v1.6.py#L962-L971)):
```python
rotation_deg = t_best.get("rotation_angle_deg", 0.0)  # e.g., 75°
rotation_rad = math.radians(rotation_deg)
half_angle = rotation_rad / 2.0

goal.pose.orientation.w = math.cos(half_angle)
goal.pose.orientation.x = 0.0
goal.pose.orientation.y = 0.0
goal.pose.orientation.z = math.sin(half_angle)
```

This quaternion encodes: **"Rotate by X degrees around Z-axis (wrist rotation)"**

### What the Robot Does

**File**: [goals.py:190-236](ur_ws_new/src/ur10e_curobo/ur10e_curobo/goals.py#L190-L236)

1. Extracts rotation angle from quaternion
2. Gets current end-effector orientation
3. Multiplies: `final_orientation = current_orientation × received_rotation`
4. Uses `final_orientation` for grasp

---

## Benefits

### ✅ Ellipse-Based Intelligence

- Uses fruit shape to determine optimal gripper width direction
- Short axis = narrow direction = less squishing

### ✅ Relative Rotation

- Works from any starting orientation
- Robot doesn't need to know absolute target orientation in world frame
- Smooth transitions between grasps

### ✅ Dense Bunch Optimization

- Adjusts ellipse orientation (±45°) to avoid neighbors
- Tests 7 angles around the base ellipse orientation
- Finds best compromise between fruit shape and neighbor avoidance

---

## Algorithm Details

### Computing Ellipse Rotation Angle

```python
# 1. Get ellipse short axis (already computed during fruit extraction)
ellipse_axis = t_best["short_axis_cam"]  # [x, y, z] in camera frame

# 2. Project onto perpendicular plane (gripper width perpendicular to approach)
approach_dir = fruit_position / ||fruit_position||
ellipse_axis_proj = ellipse_axis - dot(ellipse_axis, approach_dir) * approach_dir
ellipse_axis_proj = ellipse_axis_proj / ||ellipse_axis_proj||

# 3. Compute angle from reference direction
reference = [0, 0, 1]  # or [1, 0, 0] if approach is vertical
reference_proj = reference - dot(reference, approach_dir) * approach_dir
reference_proj = reference_proj / ||reference_proj||

angle = acos(dot(reference_proj, ellipse_axis_proj))
# Use cross product to determine sign (+/-)

# 4. For dense bunches: add ±45° adjustment
if z_std >= 0.03:
    # Test 7 angles: ellipse_angle + [-45, -30, -15, 0, 15, 30, 45]
    # Pick angle with minimum neighbor collisions
    total_angle = ellipse_angle + adjustment_angle
```

---

## Coordinate Frames

### Camera Frame (Vision Computation)

- **X**: Right
- **Y**: Down
- **Z**: Forward (depth)

Ellipse axis and rotation angles are computed in this frame.

### Robot Base Frame (Execution)

- Rotation is applied **relative to current end-effector orientation**
- No need to transform between frames (robot handles it)

---

## Visualization

The OpenCV window shows:

1. **Approach axis arrow**: Direction robot approaches from (cyan/colored)
2. **Rotation arc**: Green (positive) or magenta (negative) arc showing rotation
3. **Rotation angle text**: Shows exact angle (e.g., `+75deg`)
4. **Text in stats**: `Rot:+75deg` with color coding

**Color coding for rotation magnitude**:
- **Cyan**: 0° (no rotation from ellipse)
- **Yellow-cyan**: ≤15°
- **Orange**: ≤30°
- **Red-orange**: >30°

---

## Example Scenarios

### Scenario 1: Sparse Bunch, Ellipse at 30°

```
[ORIENT] Sparse bunch (z_std=0.0015), ellipse_rotation=+30.0°

Vision sends: rotation_angle = +30°
Robot: current_orientation × (+30° rotation) = target_orientation
Result: Gripper width aligned with ellipse short axis
```

### Scenario 2: Dense Bunch, Ellipse at 30°, Adjustment +15°

```
[ORIENT] Dense bunch (z_std=0.052), ellipse_rotation=+30.0° + adjustment=+15° = +45.0° (score=6.23)

Vision sends: rotation_angle = +45°
Robot: current_orientation × (+45° rotation) = optimized_orientation
Result: Gripper slightly rotated from ellipse to avoid neighbors
```

### Scenario 3: Dense Bunch, No Adjustment Needed

```
[ORIENT] Dense bunch (z_std=0.041), ellipse_rotation=+60.0° (no adjustment, score=2.15)

Vision sends: rotation_angle = +60°
Robot: current_orientation × (+60° rotation) = ellipse_orientation
Result: Ellipse orientation was already optimal for neighbors
```

---

## Testing

### Expected Log Output

**Sparse bunches**:
```bash
[ORIENT] Sparse bunch (z_std=0.0012), ellipse_rotation=+45.0°
```

**Dense bunches**:
```bash
[ORIENT] Dense bunch (z_std=0.048), ellipse_rotation=+30.0° + adjustment=+15° = +45.0° (score=8.45)
```

**No ellipse available**:
```bash
[ORIENT] No ellipse axis available, rotation=0°
```

### Verification

Run the system and check:

1. ✅ Ellipse-based rotation angles appear in logs
2. ✅ Different fruits show different rotation angles
3. ✅ Dense bunches show adjustment amounts
4. ✅ Visualization shows rotation arc and angle
5. ✅ Robot rotates smoothly to grasp orientation

---

## Summary

**What changed**:
- Restored ellipse short axis as **target orientation**
- Compute **rotation angle** = "how much to rotate from current to target"
- Send this angle to robot as **relative rotation**
- Robot applies rotation relative to its current pose

**Result**:
- ✅ Ellipse-based grasp intelligence (fruit shape)
- ✅ Relative rotation system (works from any starting pose)
- ✅ Dense bunch optimization (±45° adjustment)
- ✅ Smooth transitions (no absolute orientation jumps)

**Best of both worlds!** 🎯
