# main.py
import signal, sys, termios, atexit
import rclpy
from ur10e_curobo.node import UR10eCuroboMoveIt

# restore TTY on exit (even on crashes)
try:
    _fd = sys.stdin.fileno()
    _orig = termios.tcgetattr(_fd)
    atexit.register(lambda: termios.tcsetattr(_fd, termios.TCSADRAIN, _orig))
except Exception:
    pass

def _handle_signal(signum, frame):
    # try to stop nicely
    rclpy.shutdown()

signal.signal(signal.SIGINT, _handle_signal)   # Ctrl-C
signal.signal(signal.SIGTERM, _handle_signal)  # kill

def main():
    rclpy.init()
    node = UR10eCuroboMoveIt()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == "__main__":
    main()

