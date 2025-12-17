import torch
import fk

torch.cuda.init()

# Sample UR10e home/retract joint configuration
j = torch.tensor([[0.0, -2.2, 1.9, -1.383, -1.57, 0.0]], dtype=torch.float32, device='cuda')

with torch.no_grad():
    torch.cuda.synchronize()
    print("Testing forward kinematics...")

    # Option 1: direct FK
    try:
        out = fk.forward_kinematics(j)
        print("forward_kinematics() output:\n", out)
    except Exception as e:
        print("forward_kinematics() failed:", e)

    # Option 2: higher-level FK wrapper
    try:
        out2 = fk.get_end_effector_pose(j)
        print("get_end_effector_pose() output:\n", out2)
    except Exception as e:
        print("get_end_effector_pose() failed:", e)

torch.cuda.synchronize()
print("FK test complete.")

