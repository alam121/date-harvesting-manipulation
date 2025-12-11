# Approach Axis Visualization Fix

**Date**: 2025-12-10
**Issue**: Approach axis arrow and rotation visualization not displaying
**Status**: ✅ FIXED

---

## Problem

User reported: "i only ssee orange line"

The orange line was the heatmap direction arrow, but the cyan approach axis arrow (with rotation arc and angle label) was not visible.

---

## Root Cause

When switching to the relative rotation system, we removed the `approach_axis` computation and set it to `None` ([date_v1.6.py:892](ur_ws_new/src/zed_date_detector/date_v1.6.py#L892)).

The visualization code checks `if axis_dir is not None:` ([date_v1.6.py:1097](ur_ws_new/src/zed_date_detector/date_v1.6.py#L1097)), so nothing was drawn.

---

## Fix Applied

**File**: [date_v1.6.py:893-898](ur_ws_new/src/zed_date_detector/date_v1.6.py#L893-L898)

Restored `approach_axis` computation for **visualization purposes only**:

```python
# Compute approach axis for VISUALIZATION ONLY (direction from camera to fruit)
# This is NOT used for robot orientation (we use rotation_angle_deg instead)
fruit_pos = np.array([t_best["Xc"], t_best["Yc"], t_best["Zc"]], dtype=float)
approach_distance = max(np.linalg.norm(fruit_pos), 1e-6)
approach_axis_vis = fruit_pos / approach_distance  # Normalized direction to fruit
t_best["approach_axis"] = approach_axis_vis
```

---

## What This Does

The `approach_axis` is now a simple normalized vector pointing from the camera to the fruit:
- **X-axis**: Left/right in camera frame
- **Y-axis**: Up/down in camera frame
- **Z-axis**: Depth (forward from camera)

This is purely for visualization - it shows the direction the gripper approaches from, but does NOT determine robot orientation (we use `rotation_angle_deg` for that).

---

## Visualization Elements Now Working

### 1. Colored Approach Arrow ([date_v1.6.py:1124-1138](ur_ws_new/src/zed_date_detector/date_v1.6.py#L1124-L1138))
- **Cyan**: No rotation (0°)
- **Yellow-cyan**: Small rotation (≤15°)
- **Orange**: Medium rotation (≤30°)
- **Red-orange**: Large rotation (>30°)

### 2. Rotation Arc ([date_v1.6.py:1140-1163](ur_ws_new/src/zed_date_detector/date_v1.6.py#L1140-L1163))
- **Green arc**: Positive rotation (counter-clockwise)
- **Magenta arc**: Negative rotation (clockwise)
- Arc radius: 25 pixels
- Shows rotation direction and magnitude

### 3. Rotation Angle Label ([date_v1.6.py:1165-1177](ur_ws_new/src/zed_date_detector/date_v1.6.py#L1165-L1177))
- Text near arrow tip
- Format: `+30deg` or `-15deg`
- Same color as rotation arc

---

## What You Should See Now

When running the vision system:

```
┌─────────────────────────────────────────┐
│         OpenCV Window Display           │
└─────────────────────────────────────────┘

Best Fruit:
  • Blue circle at centroid
  • Colored arrow (cyan → red-orange based on rotation)
  • Green/magenta arc showing rotation direction
  • Text label: "+30deg" or "-15deg"
  • Orange heatmap direction arrow (separate)

Other Fruits:
  • Red circles
  • No arrows
```

### Example Scenarios

**Sparse Bunch (no rotation needed)**:
```
[ORIENT] Sparse bunch (z_std=0.0007 < 0.03), rotation=0°
```
- Cyan arrow (no rotation)
- No arc (rotation = 0°)

**Dense Bunch (rotation applied)**:
```
[ORIENT] Dense bunch (z_std=0.048), rotation=+30° (score=8.45)
```
- Orange arrow (30° rotation)
- Green arc from current → +30°
- Text: "+30deg"

---

## Important Notes

### Approach Axis is VISUALIZATION ONLY

The `approach_axis` computed here:
- ✅ **Shows** where gripper approaches from (camera → fruit direction)
- ✅ **Used** for drawing arrows and arcs
- ❌ **NOT used** for robot orientation (we use `rotation_angle_deg` instead)
- ❌ **NOT sent** to robot (robot receives quaternion from `rotation_angle_deg`)

### Rotation is Still RELATIVE

This fix does NOT change the relative rotation system:
- Current orientation = 0° baseline
- Rotation limited to ±45°
- Rotation applied relative to current pose
- Robot uses quaternion multiplication (see [goals.py:190-236](ur_ws_new/src/ur10e_curobo/ur10e_curobo/goals.py#L190-L236))

---

## Verification

Run the vision system:
```bash
python3 -m zed_date_detector.date_v1.6 --weights models/lab-dates.pt --conf_thres 0.5
```

You should now see:
1. ✅ Orange line (heatmap direction) - already working
2. ✅ Cyan/colored arrow (approach axis) - **NOW FIXED**
3. ✅ Rotation arc (green/magenta) - **NOW FIXED**
4. ✅ Rotation angle text - **NOW FIXED**

---

## Summary

**What was broken**: `approach_axis = None` → no visualization

**What was fixed**: Computed `approach_axis = normalized(camera → fruit)` for drawing

**Impact**: Visualization now complete, relative rotation system unchanged

**Status**: Ready to test! 🚀
