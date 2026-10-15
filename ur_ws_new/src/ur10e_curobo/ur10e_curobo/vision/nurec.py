"""Optional rectified stereo JPEG transport; SDK access stays on its owner thread."""
import copy
import math
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage


def rectified_info(camera, source_size, output_size, baseline_m=0.0):
    """ROS rectified projection: right P[3] = -fx * baseline in metres."""
    sx, sy = (output_size[i] / source_size[i] for i in range(2))
    fx, fy, cx, cy = camera.fx * sx, camera.fy * sy, camera.cx * sx, camera.cy * sy
    msg = CameraInfo()
    msg.width, msg.height = output_size
    msg.distortion_model = 'plumb_bob'
    msg.d = [0.0] * 5
    msg.k = [float(fx), 0., float(cx), 0., float(fy), float(cy), 0., 0., 1.]
    msg.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
    msg.p = [float(fx), 0., float(cx), float(-fx * baseline_m),
             0., float(fy), float(cy), 0., 0., 0., 1., 0.]
    return msg


class NuRecCapture:
    def __init__(self, node, calibration, source_size, left_frame,
                 width=1280, hz=3.0, quality=80):
        if width < 64 or not math.isfinite(hz) or not 0 < hz <= 30 or not 1 <= quality <= 100:
            raise ValueError('NuRec requires width >= 64, 0 < hz <= 30, JPEG quality 1..100')
        baseline = float(calibration.get_camera_baseline())
        # The owning ZED camera is opened with sl.UNIT.METER.
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError('NuRec requires a finite, positive SDK stereo baseline')
        width = min(width, source_size[0])
        self.size = (width, max(1, round(source_size[1] * width / source_size[0])))
        self.node, self.hz, self.quality = node, hz, quality
        self.enabled = False
        self._last_capture = float('-inf')
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.infos, self.publishers, self.topics = [], [], []
        self._bag, self.bag_path = None, None
        right_frame = left_frame.replace('left', 'right') if 'left' in left_frame else left_frame + '_right'
        for side, cam, frame, b in (
                ('left', calibration.left_cam, left_frame, 0.),
                ('right', calibration.right_cam, right_frame, baseline)):
            info = rectified_info(cam, source_size, self.size, b)
            info.header.frame_id = frame
            self.infos.append(info)
            root = '/vision/nurec/' + side
            self.topics += [root + '/image_rect/compressed', root + '/camera_info']
            self.publishers.append((
                node.create_publisher(CompressedImage, root + '/image_rect/compressed', qos),
                node.create_publisher(CameraInfo, root + '/camera_info', qos)))
        self._worker = threading.Thread(target=self._run, name='nurec_jpeg', daemon=True)
        self._worker.start()
        node.get_logger().info(
            f'[NUREC] ready (off): {self.size[0]}x{self.size[1]}, {hz:g} Hz, '
            f'JPEG {quality}, baseline={baseline:.6f} m')

    def due(self):
        now = time.monotonic()
        if not self.enabled or self._stop.is_set() or now - self._last_capture < 1. / self.hz:
            return False
        self._last_capture = now
        return True

    def submit(self, left, right, stamp):
        if not self.enabled or self._stop.is_set():
            return
        packet = (left.copy(), right.copy(), copy.deepcopy(stamp))
        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(packet)
            except queue.Full:
                pass

    def _publish(self, packet):
        left, right, stamp = packet
        messages = []
        for image, template in zip((left, right), self.infos):
            if image.shape[:2] != (self.size[1], self.size[0]):
                raise ValueError('NuRec image size does not match calibration')
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image
            ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                raise RuntimeError('NuRec JPEG encoding failed')
            info = copy.deepcopy(template)
            info.header.stamp = stamp
            msg = CompressedImage()
            msg.header = copy.deepcopy(info.header)
            msg.format = 'bgr8; jpeg compressed bgr8'
            msg.data = encoded.tobytes()
            messages.append((msg, info))
        # Encode both successfully before publishing either side.
        if self.enabled and not self._stop.is_set():
            for (image_pub, info_pub), (image, info) in zip(self.publishers, messages):
                info_pub.publish(info)
                image_pub.publish(image)

    def _run(self):
        while not self._stop.is_set():
            try:
                packet = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.enabled:
                try:
                    self._publish(packet)
                except Exception as exc:
                    self.node.get_logger().warn(f'[NUREC] pair dropped: {exc}')

    def recording(self):
        return self._bag is not None and self._bag.poll() is None

    def start_bag(self, directory='~/nurec_data/bags'):
        """Record the NuRec stereo pair to a local bag on this machine (no network
        needed, e.g. outdoors). Same topics as isaac/nurec/record_stereo.sh, plus
        TF and the URDF so replay_bag_joints.py can recover the arm pose. Default
        sqlite3 storage: the mcap plugin is not installed on the Jetson, and
        extract_frames.py reads either."""
        if self.recording():
            return self.bag_path
        path = Path(directory).expanduser() / time.strftime('nurec_%Y%m%d_%H%M%S')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(f'{path}.log', 'w') as log:
            # Same process group: a Ctrl+C on the launcher also stops the bag.
            self._bag = subprocess.Popen(
                ['ros2', 'bag', 'record', '-o', str(path), *self.topics,
                 '/tf', '/tf_static', '/robot_description'],
                stdout=log, stderr=subprocess.STDOUT)
        self.bag_path = str(path)
        self.enabled = True
        self.node.get_logger().info(f'[NUREC] recording bag -> {path}')
        return self.bag_path

    def stop_bag(self):
        """SIGINT lets rosbag2 finish metadata.yaml; reaped off-thread."""
        proc, self._bag = self._bag, None
        self.enabled = False
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            threading.Thread(target=proc.wait, daemon=True).start()
            self.node.get_logger().info(f'[NUREC] bag stopped: {self.bag_path}')
        return proc is not None

    def close(self):
        if self.recording():
            self._bag.send_signal(signal.SIGINT)
            try:
                self._bag.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                self._bag.kill()
        self.enabled = False
        self._stop.set()
        self._worker.join(timeout=2.)
