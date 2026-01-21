# TROUBLESHOOTING.md

**Project:** ManipulatorsDatePalm – Jetson + cuRobo + UR10e + ZED + YOLO

This document collects the most common runtime, build, and deployment issues observed on the UR10e + cuRobo + ZED stack and their fixes.

---

## Table of Contents

1. [cuRobo install fails: libcudss.so.0](#1-curobo-install-fails-libcudssso0)
2. [cuRobo build fails: unsupported GPU architecture](#2-curobo-build-fails-unsupported-gpu-architecture-jetson)
3. [cuRobo build fails: ninja not found](#3-curobo-build-fails-ninja-not-found)
4. [PyTorch import fails or segfaults](#4-pytorch-import-fails-or-segfaults)
5. [ZED SDK segfaults at runtime](#5-zed-sdk-segfaults-at-runtime)
6. [ZED TF transform fails](#6-zed-tf-transform-fails)
7. [UR10e robot not connecting](#7-ur10e-robot-not-connecting)
8. [Delto gripper not responding](#8-delto-gripper-not-responding)
9. [YOLO TensorRT engine fails to load](#9-yolo-tensorrt-engine-fails-to-load)
10. [RViz fails on headless Jetson](#10-rviz-fails-on-headless-jetson)
11. [ROS 2 nodes silently die](#11-ros-2-nodes-silently-die)
12. [Debug Checklist](#debug-checklist-run-this-first)
13. [Key Lessons](#key-lessons)
14. [When to Escalate](#when-to-escalate)

---

## 1) cuRobo install fails: `ImportError: libcudss.so.0`

### Symptom
```text
ImportError: libcudss.so.0: cannot open shared object file
pip install -e . --no-build-isolation
metadata-generation-failed
```

##Cause
PyTorch depends on NVIDIA cuDSS. cuDSS is installed but placed in a non-standard directory that is not searched by the dynamic linker.

Fix
```bash
sudo apt install -y libcudss0-cuda-12 libcudss0-dev-cuda-12
echo "/usr/lib/aarch64-linux-gnu/libcudss/12" | sudo tee /etc/ld.so.conf.d/libcudss.conf
sudo ldconfig
```
Verify
```bash
ldconfig -p | grep cudss
python3 -c "import torch; print(torch.__version__)"
```

## 2) cuRobo install fails: `canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'`

### Symptom
```text
TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
Preparing editable metadata (pyproject.toml) ... error
error: metadata-generation-failed
```

### Cause

Python packaging toolchain mismatch.
setuptools calls canonicalize_version(..., strip_trailing_zero=...) but the installed packaging version is too old and does not support this argument.
This typically happens when mixing Ubuntu system Python packages with user-installed pip packages.

### Fix
```text
python3 -m pip install --user -U pip setuptools wheel packaging setuptools-scm
```

### Verify
```text
python3 -c "import pip, setuptools, packaging; print(pip.__version__, setuptools.__version__, packaging.__version__)"
python3 -m pip install -e . --no-build-isolation -v
```
