# main.py
import os, signal, sys, termios, atexit
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

# Python hands the GIL between threads at most once per "switch interval",
# default 5ms. cuRobo's IK runs many small CUDA syncs inside a single
# solve_single(); each releases the GIL and must then reacquire it, queueing
# behind this node's 4 executor threads and its 40-50Hz timers. Measured in the
# field: 22-28ms of thread CPU stretched across 182-244ms of wall clock -- the
# thread was descheduled ~90% of the time, which is why the same call
# benchmarks at 14ms in an idle process. A shorter interval hands the GIL back
# sooner; the cost is slightly more context switching.
#
# Measured on this node, per corridor-preflight solve (wall vs thread CPU):
#   5ms  (python default)  call 182-244ms, cpu 22-28ms   ik_preflight 953ms
#   0.5ms                  call  82-164ms, cpu 16-25ms   ik_preflight 717-744ms
#   0.1ms (current)        call  61-79ms,  cpu 17-20ms   ik_preflight 289-305ms
#
# 3.2x on the preflight, and the variance collapsed too: the spread went from
# 82/316/137/147ms to 79/83/68/71ms, because the outliers were handoff stalls
# rather than slow solves. Cycle time 23.6s -> 19.0s. Still ~3.7x starved
# against the 14ms this call benchmarks at in an idle process; closing that
# last gap needs planning moved off the ROS executor, not a smaller interval.
#
# Tune or disable with UR10E_GIL_SWITCH_INTERVAL (seconds; <=0 leaves the
# default in place). Lower is not free: the interpreter checks for thread
# switches that much more often, which taxes every other Python thread in the
# node, so back off to 0.0002 if callbacks or the GUI start to feel sluggish.
_GIL_SWITCH_INTERVAL = float(os.environ.get("UR10E_GIL_SWITCH_INTERVAL", "0.0001"))
if _GIL_SWITCH_INTERVAL > 0:
    sys.setswitchinterval(_GIL_SWITCH_INTERVAL)


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