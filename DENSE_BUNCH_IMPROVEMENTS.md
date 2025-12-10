# Dense Bunch Grasp Improvements

## Overview

This document describes the improvements made to handle **dense date bunch grasping** where fruits are tightly packed together. The main issue was that the gripper would stop closing when **one finger contacted a neighbor fruit**, resulting in weak or failed grasps.

**Success Rate Improvement**: Expected to increase from **~74%** to **~90%+** on dense bunches.

---

## Problem Analysis

### Original Behavior
- **Single global force threshold** (`-0.15`) applied to all 3 fingers
- Gripper stopped closing when **ALL** fingers detected contact
- In dense bunches: one finger hits neighbor fruit → premature stop → weak grip

### Root Cause
```
Dense Bunch Scenario:
  ┌─────────────────────────────┐
  │  🍇  ← target fruit         │
  │  🍇🍇  ← neighbor fruits     │
  └─────────────────────────────┘

Gripper approaches:
  Left finger → hits neighbor (early contact at -0.16)
  Center finger → contacts target (at -0.20)
  Right finger → contacts target (at -0.17)

Old logic: ALL 3 must contact → stops at -0.16 → WEAK GRIP ❌
New logic: MINIMUM 2 must contact → continues to -0.20 → PROPER GRIP ✅
```

---

## Solution Implemented

### 1. **Per-Finger Force Thresholds**
Instead of a single threshold, each finger now has independent sensitivity:

```python
# In config.py
force_threshold_left: float = -0.14    # finger 0 (softer)
force_threshold_center: float = -0.16  # finger 1 (stronger)
force_threshold_right: float = -0.14   # finger 2 (softer)
```

**Why different values?**
- **Center finger** contacts first in suction mode (higher threshold)
- **Left/Right fingers** are slightly softer to avoid false positives

---

### 2. **Minimum Fingers for Stop**
New parameter: `min_fingers_for_stop` (default: **2**)

```python
# In delto_gripper_controller.py
contacted_count = sum(1 for i in (0, 1, 2) if self.channel_contact(i))
threshold_reached = contacted_count >= self.min_fingers_for_stop
```

**Behavior**:
- `min_fingers_for_stop = 2`: Accept partial contact (dense bunches) ✅
- `min_fingers_for_stop = 3`: Require all fingers (isolated fruits)

---

### 3. **Enhanced Logging**
When partial contact is detected, the system now logs:

```
🔶 Partial contact: 2/3 fingers contacted [0, 1] (forces: ['-0.16', '-0.20', '-0.10'])
```

This helps you:
- Debug grasp failures
- Tune thresholds based on real force data
- Identify which finger is hitting neighbors

---

## Configuration

All parameters are configurable via **ROS parameters** or **config.py**:

### Via ROS Parameters (Runtime)
```bash
ros2 run ur10e_curobo main \
  --ros-args \
  -p gripper.force_threshold_left:=-0.14 \
  -p gripper.force_threshold_center:=-0.16 \
  -p gripper.force_threshold_right:=-0.14 \
  -p gripper.min_fingers_for_stop:=2 \
  -p gripper.closing_steps:=10 \
  -p gripper.step_delay_s:=0.2 \
  -p gripper.use_suction:=false
```

### Via config.py (Code Defaults)
```python
# In ur10e_curobo/config.py
@dataclass
class Gripper:
    force_threshold_left: float = -0.14
    force_threshold_center: float = -0.16
    force_threshold_right: float = -0.14
    min_fingers_for_stop: int = 2
    closing_steps: int = 10
    step_delay_s: float = 0.2
    use_suction: bool = False
```

---

## Tuning Guide

### For Dense Bunches (Current Focus)
```python
min_fingers_for_stop = 2  # Accept partial contact
force_threshold_left = -0.13  # Lower if left finger too sensitive
force_threshold_center = -0.16  # Baseline
force_threshold_right = -0.13  # Lower if right finger too sensitive
```

### For Isolated Fruits (High Precision)
```python
min_fingers_for_stop = 3  # Require all fingers
force_threshold_left = -0.15
force_threshold_center = -0.17
force_threshold_right = -0.15
```

### If Gripper Closes Too Weakly
**Symptom**: Classifier reports "WEAK" or "SLIPPED"

**Solution**: Lower thresholds (more negative = tighter grip)
```python
force_threshold_left = -0.15  # was -0.14
force_threshold_center = -0.17  # was -0.16
force_threshold_right = -0.15  # was -0.14
```

### If Gripper Stops Too Early
**Symptom**: Gripper doesn't close fully, log shows "Partial contact: 1/3"

**Solution**:
1. Keep `min_fingers_for_stop = 2` (or lower to 1 temporarily)
2. Increase threshold for the problematic finger (less sensitive)
```python
# If left finger (0) hits early:
force_threshold_left = -0.12  # was -0.14 (less sensitive)
```

---

## Testing Procedure

### 1. Baseline Test (Isolated Fruits)
```bash
# Use strict settings
-p gripper.min_fingers_for_stop:=3
```
- Should maintain **99% success** on small bunches
- Confirms thresholds are correct

### 2. Dense Bunch Test
```bash
# Use relaxed settings
-p gripper.min_fingers_for_stop:=2
```
- Monitor logs for partial contact messages
- Track success rate improvement

### 3. Force Data Collection
Add this to your test script to collect force signatures:
```python
# After each grasp attempt
forces = node.gripper_controller.force_data
outcome = node.last_grasp_end_template  # PROPER, WEAK, etc.
print(f"Outcome: {outcome}, Forces: {forces}")
```

**Save to file for analysis:**
```bash
# Run tests and save logs
ros2 run ur10e_curobo main 2>&1 | tee grasp_log.txt

# Extract force data
grep "Partial contact" grasp_log.txt > force_analysis.txt
```

---

## Expected Results

### Before (Original Code)
| Scenario | Success Rate |
|----------|--------------|
| Small bunches (isolated fruits) | 99% |
| Dense bunches (3+ fruits touching) | ~74% |

**Failure mode**: Gripper stops at first contact → weak grip → slip during retraction

### After (With Improvements)
| Scenario | Success Rate |
|----------|--------------|
| Small bunches | 99% (unchanged) |
| Dense bunches | **~90%+** (expected) |

**Improvement**: Gripper continues closing even when one finger contacts neighbor

---

## Code Changes Summary

### Modified Files
1. **[delto_gripper_controller.py](ur_ws_new/src/ur10e_curobo/ur10e_curobo/delto_gripper_controller.py)**
   - Added per-finger thresholds
   - Implemented minimum fingers logic
   - Enhanced logging

2. **[config.py](ur_ws_new/src/ur10e_curobo/ur10e_curobo/config.py)**
   - Added `Gripper` dataclass with all tunable parameters

3. **[gripper.py](ur_ws_new/src/ur10e_curobo/ur10e_curobo/gripper.py)**
   - Updated `init_gripper()` to pass config values

4. **[node.py](ur_ws_new/src/ur10e_curobo/ur10e_curobo/node.py)**
   - Added ROS parameter declarations for gripper config
   - Read parameters and pass to gripper controller

### Key Functions
- `DeltoGripperController.channel_contact(i)`: Per-finger threshold check
- `DeltoGripperController.is_force_threshold_reached()`: Minimum fingers logic
- `init_gripper()`: Configuration integration

---

## Future Improvements (Not Yet Implemented)

### 1. **Vision-Based Occlusion Scoring**
Add penalty for occluded fruits in target selection:
```python
# In date_v1.6.py
score = dist + (z_std * 10.0) + edge_penalty
```

### 2. **Two-Stage Closing**
Add "squeeze phase" after initial contact:
```python
# Stage 1: Approach (current)
# Stage 2: Squeeze harder on contacted fingers
```

### 3. **ML-Based Grasp Classifier**
Replace hardcoded templates with trained model:
- Collect force data from 100+ grasps
- Train Random Forest classifier
- Handle asymmetric contact patterns

### 4. **Pre-Grasp Isolation Move**
Wiggle gripper to push neighbor fruits aside before closing

---

## Troubleshooting

### Issue: Gripper doesn't close at all
**Cause**: Thresholds too low (too sensitive)

**Fix**: Increase thresholds (less negative):
```python
force_threshold_left = -0.12  # was -0.14
```

### Issue: Still getting slips on dense bunches
**Possible causes**:
1. `min_fingers_for_stop` still too high → lower to 1 temporarily
2. Vision picking occluded fruits → implement occlusion scoring
3. Fruit too slippery → increase closing steps for tighter grip

**Debug steps**:
```bash
# Check which fingers are contacting
grep "Partial contact" logs.txt

# If seeing "1/3 fingers contacted" → lower min_fingers_for_stop
# If seeing "3/3 fingers contacted" but still slips → lower thresholds
```

### Issue: Gripper crushing fruits
**Cause**: Thresholds too high (too aggressive)

**Fix**: Raise thresholds (less negative):
```python
force_threshold_center = -0.14  # was -0.16
```

---

## Performance Monitoring

### Metrics to Track
```python
# Add to your test harness
total_attempts = 0
successes = 0
partial_contacts = 0
full_contacts = 0

# After each grasp:
if contacted_count == 2:
    partial_contacts += 1
elif contacted_count == 3:
    full_contacts += 1

if outcome == "GRABBED":
    successes += 1

success_rate = successes / total_attempts
partial_rate = partial_contacts / total_attempts
```

**Target KPIs**:
- Success rate on dense bunches: **>85%**
- Partial contact rate: **30-50%** (normal for dense bunches)
- Average force variance: **<0.03** (stable grip)

---

## Contact & Support

For questions or issues:
1. Check logs for "Partial contact" messages
2. Collect force data using the procedure above
3. Adjust thresholds incrementally (±0.01 at a time)
4. Test on 20+ attempts before concluding

**Good luck with your dense bunch testing! 🍇🤖**
