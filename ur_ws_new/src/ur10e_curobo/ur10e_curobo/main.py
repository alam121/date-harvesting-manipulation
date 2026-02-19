# main.py
import signal, sys, termios, atexit
import rclpy
from rclpy.executors import MultiThreadedExecutor
from ur10e_curobo.node import UR10eCuroboMoveIt

# restore TTY on exit (even on crashes)
try:
    _fd = sys.stdin.fileno()
    _orig = termios.tcgetattr(_fd)
    atexit.register(lambda: termios.tcsetattr(_fd, termios.TCSADRAIN, _orig))
except Exception:
    pass

exit_signal = False
# ===========================

def _handle_signal(signum, frame):
    global exit_signal  # Declare that you are modifying the global variable
    exit_signal = True
    print("\n[Signal] Graceful shutdown requested.")
    # We removed rclpy.shutdown() from here, which is correct.

signal.signal(signal.SIGINT, _handle_signal)   # Ctrl-C
signal.signal(signal.SIGTERM, _handle_signal)  # kill

def main():
    rclpy.init()
    node = UR10eCuroboMoveIt()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        while rclpy.ok() and not exit_signal:
            executor.spin_once(timeout_sec=0.1)

    finally:
        node.stop_requested = True
        try:
            executor.shutdown()
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        print("[System] Shutdown complete.")

if __name__ == "__main__":
    main()