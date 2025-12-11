# Relative Rotation from Ellipse Orientation - Implementation

**Date**: 2025-12-11
**Status**: ✅ Implemented

---

## Overview

The system now computes **relative rotation** from the robot's current orientation to the ellipse-based target orientation.

- **Ellipse short axis** = Target orientation (what we want)
- **Current gripper orientation** = Starting point (0° baseline)
- **Rotation angle sent to robot** = Difference between current and target

---

## Key Concept

The vision system:
1. Extracts ellipse short axis (fruit's narrow direction) ✅ Verified correct
2. For dense bunches: rotates axis ±45° to avoid neighbors
3. Transforms axis from camera frame to base frame
4. Gets current gripper orientation from TF
5. Computes angle between current gripper Y-axis and target axis
6. Sends this **relative rotation angle** to robot

The robot:
1. Receives relative rotation angle
2. Applies rotation FROM current orientation TO target orientation
3. Uses this for grasp phase (two-phase strategy)

---

## What Changed

### 1. Verified Ellipse Extraction

**File**: [date_v1.6.py:643-682](ur_ws_new/src/zed_date_detector/date_v1.6.py#L643-L682)

✅ **Confirmed**: The code correctly extracts the **SHORT axis** (perpendicular to major axis)

```python
# Line 658: cv2.fitEllipse returns (major, minor), angle_deg
ellipse = cv2.fitEllipse(cnt)
(xc2d, yc2d), (major, minor), angle_deg = ellipse

# Line 661-664: Compute LONG axis direction from angle
long_dir_img = np.array([math.cos(theta), math.sin(theta)])

# Line 666-669: Compute SHORT axis = perpendicular to long
short_dir_cam = np.array([-long_dir_img[1], long_dir_img[0], 0.0])
```

This is **correct** - the short axis is the perpendicular to the major axis, which is the narrow direction of the fruit.

---

### 2. Refactored `optimize_grasp_orientation_for_dense_bunch()`

**File**: [date_v1.6.py:144-267](ur_ws_new/src/zed_date_detector/date_v1.6.py#L144-L267)

**Before**: Returned rotation angle in camera frame (wrong - arbitrary reference)

**After**: Returns optimized **axis direction** in camera frame

```python
# For sparse bunches: return ellipse axis as-is
if z_std < z_std_threshold:
    return ellipse_axis_proj  # [x, y, z] in camera frame

# For dense bunches: return ellipse axis rotated by ±45° adjustment
optimized_axis = (
    ellipse_axis_proj * cos_a +
    np.cross(approach_dir, ellipse_axis_proj) * sin_a +
    approach_dir * np.dot(approach_dir, ellipse_axis_proj) * (1 - cos_a)
)
return optimized_axis  # [x, y, z] in camera frame
```

**Key change**: No longer computing angles with arbitrary reference - just returns the 3D axis direction.

---

### 3. Implemented Relative Rotation Computation

**File**: [date_v1.6.py:915-992](ur_ws_new/src/zed_date_detector/date_v1.6.py#L915-L992)

**New code** that:

1. **Gets current gripper orientation from TF**:
```python
transform = tf_buffer.lookup_transform("base_link", "gripper_tip", rclpyTime())
current_quat = transform.transform.rotation
current_rot = R.from_quat([...])
current_matrix = current_rot.as_matrix()
```

2. **Transforms target axis from camera to base frame**:
```python
# Transform axis direction vector
axis_point_cam = PointStamped()
axis_point_cam.header.frame_id = cam_frame
axis_point_cam.point.x/y/z = optimized_axis_cam[0/1/2]

axis_point_base = tf_buffer.transform(axis_point_cam, "base_link")
target_axis_base = np.array([...])  # Direction in base frame
```

3. **Computes angle between current and target**:
```python
# Current gripper Y-axis (width direction)
gripper_y_current = current_matrix[:, 1]

# Project both onto plane perpendicular to approach
target_proj = target_axis_base - dot(...) * gripper_z_current
gripper_y_proj = gripper_y_current - dot(...) * gripper_z_current

# Compute angle
dot = np.dot(gripper_y_proj, target_proj)
angle_rad = math.acos(dot)

# Determine sign
cross = np.cross(gripper_y_proj, target_proj)
if np.dot(cross, gripper_z_current) < 0:
    angle_rad = -angle_rad
```

4. **Sends relative rotation angle**:
```python
relative_angle_deg = math.degrees(angle_rad)
t_best["rotation_angle_deg"] = relative_angle_deg
print(f"[ORIENT] Relative rotation: {relative_angle_deg:+.1f}° (current→target)")
```

---

## How It Works End-to-End

### Sparse Bunch Example

```
1. Vision detects fruit
2. Ellipse fitting: short axis = [0.3, 0.7, 0.1] in camera frame
3. optimize_grasp_orientation_for_dense_bunch() returns [0.3, 0.7, 0.1]
4. Transform to base frame: [0.5, -0.6, 0.2]
5. Current gripper Y-axis: [0.8, -0.5, 0.1] in base frame
6. Compute angle: 15° rotation needed
7. Send rotation_angle_deg = 15°
8. Robot rotates FROM current BY 15° TO align with ellipse
```

**Result**: If current gripper orientation already matches ellipse → rotation = 0°!

---

### Dense Bunch Example

```
1. Vision detects fruit in dense bunch (z_std = 0.045)
2. Ellipse fitting: short axis = [0.3, 0.7, 0.1]
3. Test ±45° adjustments:
   - 0°: collision_score = 12.5
   - +30°: collision_score = 8.2  ← Best!
   - -15°: collision_score = 10.1
4. Rotate ellipse axis by +30°: optimized_axis = [0.5, 0.6, 0.1]
5. Transform to base frame: [0.6, -0.5, 0.15]
6. Current gripper Y-axis: [0.8, -0.5, 0.1]
7. Compute angle: 22° rotation needed
8. Send rotation_angle_deg = 22°
9. Robot rotates FROM current BY 22° TO optimized orientation
```

**Result**: Ellipse orientation + dense bunch adjustment, relative to current pose!

---

## Expected Log Output

### Sparse Bunch
```bash
[ORIENT] Sparse bunch (z_std=0.0012), using ellipse axis
[ORIENT] Relative rotation: +15.3° (current→target)
```

### Dense Bunch (No Adjustment)
```bash
[ORIENT] Dense bunch (z_std=0.048), using ellipse axis (no adjustment, score=3.21)
[ORIENT] Relative rotation: +8.7° (current→target)
```

### Dense Bunch (With Adjustment)
```bash
[ORIENT] Dense bunch (z_std=0.052), ellipse axis + +30° adjustment (score=7.45)
[ORIENT] Relative rotation: +22.1° (current→target)
```

### No Ellipse Available
```bash
[ORIENT] No ellipse axis available
[ORIENT] No target axis, rotation=0° (keep current)
```

### Current Orientation Already Matches Target
```bash
[ORIENT] Sparse bunch (z_std=0.0008), using ellipse axis
[ORIENT] Relative rotation: +0.2° (current→target)
```
→ Should be close to 0° if gripper is already aligned!

---

## Benefits

### ✅ True Relative Rotation
- Rotation angle is **relative to current gripper orientation**
- 0° means "keep current orientation" (gripper already aligned)
- Non-zero means "rotate by this amount FROM current TO target"

### ✅ Ellipse-Based Intelligence
- Short axis = narrow direction = less squishing
- Works for all fruit shapes and orientations

### ✅ Dense Bunch Optimization
- ±45° adjustment around ellipse to avoid neighbors
- Balances fruit shape with collision avoidance

### ✅ No Arbitrary References
- No more angles computed from arbitrary [0, 0, 1] reference
- All angles are **meaningful**: current → target

---

## Testing

### Verification Steps

1. **Check ellipse extraction**:
   - Should see `[ORIENT] Sparse bunch...using ellipse axis` or `Dense bunch...`

2. **Check relative rotation**:
   - Should see `[ORIENT] Relative rotation: X° (current→target)`
   - Angle should change based on current gripper orientation

3. **Test with aligned gripper**:
   - Manually align gripper with fruit
   - Rotation angle should be close to 0°

4. **Test with rotated gripper**:
   - Rotate gripper 90° from fruit orientation
   - Rotation angle should be close to 90°

### Expected Behavior

**Scenario 1**: Gripper already aligned with fruit
```
[ORIENT] Relative rotation: +0.5° (current→target)
```
→ Small angle, robot barely rotates

**Scenario 2**: Gripper perpendicular to fruit
```
[ORIENT] Relative rotation: +87.2° (current→target)
```
→ Large angle, robot rotates significantly

**Scenario 3**: Dense bunch with adjustment
```
[ORIENT] Dense bunch, ellipse axis + +30° adjustment
[ORIENT] Relative rotation: +45.3° (current→target)
```
→ Combined effect: rotation to ellipse + neighbor avoidance

---

## Coordinate Frames

### Camera Frame (Vision Computation)
- **X**: Right
- **Y**: Down
- **Z**: Forward (depth)

Ellipse axis extracted in this frame.

### Base Frame (Robot Execution)
- Transform: `tf_buffer.transform(axis_cam, "base_link")`
- Rotation computed in this frame
- Current gripper orientation queried in this frame

### Gripper Frame (Current Orientation)
- **Y-axis**: Gripper width direction (what we align with ellipse)
- **Z-axis**: Approach direction (gripper fingers extend along this)

---

## Dependencies Added

```python
from scipy.spatial.transform import Rotation as R
```

**Why**: Needed for quaternion operations and rotation matrix extraction.

---

## Files Modified

1. **[date_v1.6.py](ur_ws_new/src/zed_date_detector/date_v1.6.py)**
   - Line 26: Added scipy import
   - Line 144-267: Refactored `optimize_grasp_orientation_for_dense_bunch()` to return axis direction
   - Line 896-901: Changed function call to get optimized axis
   - Line 915-992: Implemented relative rotation computation using TF

2. **[goals.py](ur_ws_new/src/ur10e_curobo/ur10e_curobo/goals.py)** (no changes needed)
   - Already implements relative rotation on robot side (lines 190-236)

---

## Summary

**What changed**:
1. ✅ Verified ellipse extraction gets SHORT axis (correct)
2. ✅ Refactored optimization function to return axis direction (not angle)
3. ✅ Implemented true relative rotation using gripper TF
4. ✅ Rotation angle = angle FROM current TO target (ellipse-based)

**Result**:
- **Ellipse orientation** = Target (what we want)
- **Current orientation** = Baseline (0° reference)
- **Rotation sent** = Difference (how much to rotate)
- **Robot behavior** = Smooth relative rotation to optimal orientation

**Ready to test!** 🚀
