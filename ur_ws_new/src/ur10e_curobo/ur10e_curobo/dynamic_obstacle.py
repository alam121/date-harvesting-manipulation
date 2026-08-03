from curobo.geom.types import Sphere, Cuboid
from visualization_msgs.msg import Marker, MarkerArray

class DynamicObstacleManager:
    def __init__(self, node, motion_gen, world_model):
        self.node = node
        self.motion_gen = motion_gen
        self.world_model = world_model

        # Publish all markers as MarkerArray
        self.marker_pub = node.create_publisher(MarkerArray, "/curobo_dynamic_obstacles", 10)

        # Timer to refresh RViz continuously
        self.timer = node.create_timer(0.1, self.publish_all_markers)

    # ------------------------------
    # Add Sphere
    # ------------------------------
    def _try_update_world(self):
        if getattr(self.node, "_cuda_faulted", False):
            return
        _yolo = getattr(self.node, 'yolo_thread', None)
        _yolo_lock = getattr(_yolo, 'inference_lock', None)
        if _yolo_lock: _yolo_lock.acquire()
        try:
            self.motion_gen.update_world(self.world_model)
        except Exception as e:
            self.node.get_logger().warn(f"[DynamicObstacleManager] update_world failed: {e}")
            if "CUDA error" in str(e) or "illegal memory access" in str(e):
                self.node._cuda_faulted = True
        finally:
            if _yolo_lock: _yolo_lock.release()

    def add_sphere(self, name, radius=0.1, initial_pose=[0,0,-10,1,0,0,0]):
        sphere = Sphere(name=name, pose=initial_pose, radius=radius)
        self.world_model.sphere.append(sphere)
        self._try_update_world()
        self.node.get_logger().info(f"[DynamicObstacleManager] Added sphere '{name}'")

    # ------------------------------
    # Add Cuboid
    # ------------------------------
    def add_cuboid(self, name, dims, initial_pose=[0,0,-10,1,0,0,0]):
        cube = Cuboid(name=name, pose=initial_pose, dims=dims)
        self.world_model.cuboid.append(cube)
        self._try_update_world()
        self.node.get_logger().info(f"[DynamicObstacleManager] Added cuboid '{name}'")

    # ------------------------------
    # Update Pose (Dynamic)
    # ------------------------------
    def update_pose(self, name, position):
        updated = False

        # Update spheres
        for s in self.world_model.sphere:
            if s.name == name:
                s.pose = [position[0], position[1], position[2], 1.0, 0.0, 0.0, 0.0]
                updated = True

        # Update cuboids
        for c in self.world_model.cuboid:
            if c.name == name:
                c.pose = [position[0], position[1], position[2], 1.0, 0.0, 0.0, 0.0]
                updated = True

        if updated:
            self._try_update_world()
        else:
            self.node.get_logger().warn(f"[DynamicObstacleManager] Obstacle '{name}' not found!")

    # ------------------------------
    # Publish all markers to RViz
    # ------------------------------
    def publish_all_markers(self):
        msg = MarkerArray()
        t = self.node.get_clock().now().to_msg()

        # Publish spheres
        for idx, s in enumerate(self.world_model.sphere):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = t
            marker.ns = "dynamic_obstacles"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(s.pose[0])
            marker.pose.position.y = float(s.pose[1])
            marker.pose.position.z = float(s.pose[2])

            marker.pose.orientation.w = 1.0  # No rotation needed

            marker.scale.x = s.radius * 2
            marker.scale.y = s.radius * 2
            marker.scale.z = s.radius * 2

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker.lifetime.sec = 0  # infinite lifetime
            msg.markers.append(marker)

        # Publish cuboids
        for idx, c in enumerate(self.world_model.cuboid, start=1000):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = t
            marker.ns = "dynamic_obstacles"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = c.pose[0]
            marker.pose.position.y = c.pose[1]
            marker.pose.position.z = c.pose[2]

            marker.pose.orientation.w = 1.0

            marker.scale.x = c.dims[0]
            marker.scale.y = c.dims[1]
            marker.scale.z = c.dims[2]

            marker.color.r = 0.0
            marker.color.g = 0.3
            marker.color.b = 1.0
            marker.color.a = 0.8

            marker.lifetime.sec = 0
            msg.markers.append(marker)

        self.marker_pub.publish(msg)

    # Debug Print
    # ------------------------------
    def print_world(self):
        print("========== CURRENT CUROBO WORLD ==========")

        print("\nSPHERES:")
        if len(self.world_model.sphere) == 0:
            print("  (none)")
        else:
            for s in self.world_model.sphere:
                print(f"  - name={s.name}, pose={s.pose}, radius={s.radius}")

        print("\nCUBOIDS:")
        if len(self.world_model.cuboid) == 0:
            print("  (none)")
        else:
            for c in self.world_model.cuboid:
                print(f"  - name={c.name}, dims={c.dims}, pose={c.pose}")

        print("\n============================================\n")

    def ask_user_position(self, name):
        print(f"Enter new XYZ for obstacle '{name}'")

        try:
            x = float(input("X: "))
            y = float(input("Y: "))
            z = float(input("Z: "))
        except ValueError:
            print("[DynamicObstacleManager] Invalid input!")
            return None

        return [x, y, z]
