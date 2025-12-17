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
    try:
        # This loop (line 27) will now work because 'exit_signal' exists
        while rclpy.ok() and not exit_signal:
            rclpy.spin_once(node, timeout_sec=0.1)

        # IMPORTANT: Add your thread cleanup logic here before the 'finally' block
        # (This is the next step to stop Thread-1 from crashing)
        # e.g., node.wait_for_threads() 

    finally:
        # This block runs when the 'while' loop exits (or if an error occurs)
        node.stop_requested = True
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        print("[System] Shutdown complete.")

if __name__ == "__main__":
    main()