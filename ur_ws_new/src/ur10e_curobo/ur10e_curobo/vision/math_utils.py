"""Math and geometry utility functions for vision module."""

import math
import numpy as np


def unit_vector(v):
    """Normalize a vector to unit length."""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def quat_mul(q, r):
    """Multiply two quaternions (w, x, y, z)."""
    w, x, y, z = q
    W, X, Y, Z = r
    return (
        w * W - x * X - y * Y - z * Z,
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
    )


def quat_rotate_vec(q, v3):
    """Rotate vector v3 by quaternion q (w, x, y, z)."""
    qv = (0.0, v3[0], v3[1], v3[2])
    qi = (q[0], -q[1], -q[2], -q[3])
    return quat_mul(quat_mul(q, qv), qi)[1:]


def quat_align_x_to_axis(axis_world, up_hint=(0, 0, 1)):
    """
    Build quaternion that maps robot's +X axis to the given axis_world,
    with up_hint used to resolve roll.
    """
    X = unit_vector(axis_world)
    U = unit_vector(up_hint)
    Y = np.cross(U, X)
    if np.linalg.norm(Y) < 1e-6:
        U = np.array([1.0, 0.0, 0.0])
        Y = np.cross(U, X)
    Y = unit_vector(Y)
    Z = unit_vector(np.cross(X, Y))

    R = np.array(
        [
            [X[0], Y[0], Z[0]],
            [X[1], Y[1], Z[1]],
            [X[2], Y[2], Z[2]],
        ],
        dtype=float,
    )
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return (float(qw), float(qx), float(qy), float(qz))


def ray_sphere_intersection(ray_origin, ray_dir, sphere_center, sphere_radius):
    """
    Check if a ray intersects a sphere.

    Returns:
        (hit, t_enter) where t_enter is distance along ray to first intersection.
        Returns (False, inf) if no intersection.
    """
    oc = ray_origin - sphere_center
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4.0 * c

    if discriminant < 0:
        return False, float('inf')

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    if t1 > 0.001:
        return True, t1
    elif t2 > 0.001:
        return True, t2
    return False, float('inf')


def project_point_to_image(pt_cam, intrinsics, scale):
    """Project a 3D point in the camera frame to display pixel coords."""
    X, Y, Z = pt_cam
    if Z <= 0 or not np.isfinite(Z):
        return None
    u = intrinsics["fx"] * X / Z + intrinsics["cx"]
    v = intrinsics["fy"] * Y / Z + intrinsics["cy"]
    return (int(u * scale[0]), int(v * scale[1]))
