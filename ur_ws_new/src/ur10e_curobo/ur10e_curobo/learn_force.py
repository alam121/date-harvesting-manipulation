#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import csv
import time
import os
from datetime import datetime
import threading


class ForceDataCollector(Node):
    def __init__(self, output_dir: str = None):
        super().__init__('force_data_collector')

        if output_dir is None:
            output_dir = os.path.expanduser('~/force_data')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(output_dir, f'force_data_{timestamp}.csv')

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'elapsed_s', 'f0', 'f1', 'f2', 'label'])

        self.start_time = time.time()
        self.current_label = 'idle'
        self.sample_count = 0
        self.recording = False
        self.waiting_for_user = False   # <-- prevents double input

        self.force_sub = self.create_subscription(
            Float32MultiArray,
            '/gripper/force',
            self.force_callback,
            10
        )

        self.print_commands()

    def print_commands(self):
        print("─────────────────────────────────────────────")
        print("Commands:")
        print("  r  → start recording")
        print("  s  → stop recording")
        print("  q  → quit")
        print("Labels: open, close_nothing, grabbed")
        print("─────────────────────────────────────────────\n")

    # ------------------------------------------------------------------
    # FORCE CALLBACK (manual sample confirmation)
    # ------------------------------------------------------------------
    def force_callback(self, msg):
        if not self.recording:
            return
        if self.waiting_for_user:
            return  # Prevent overlapping prompts

        forces = list(msg.data)[:3]
        now = time.time()
        elapsed = now - self.start_time

        # Show sample
        print("\n-----------------------------------------")
        print("New sample:")
        print(f"  f0={forces[0]:.3f}, f1={forces[1]:.3f}, f2={forces[2]:.3f}")
        print(f"Current label: {self.current_label}")
        print("Save?  y/n   or change label: o=open, c=close_nothing, g=grabbed")
        print("-----------------------------------------")

        # Mark that we're expecting input
        self.waiting_for_user = True

        # Ask in a separate thread to avoid blocking ROS
        threading.Thread(
            target=self.get_user_input,
            args=(now, elapsed, forces),
            daemon=True
        ).start()

    # ------------------------------------------------------------------
    # HANDLE USER INPUT FOR SAMPLE
    # ------------------------------------------------------------------
    def get_user_input(self, timestamp, elapsed, forces):
        user = input(">>> ").strip().lower()

        if user == 'y':
            self.csv_writer.writerow([
                timestamp,
                round(elapsed, 4),
                round(forces[0], 4),
                round(forces[1], 4),
                round(forces[2], 4),
                self.current_label
            ])
            self.sample_count += 1
            print(f"[Saved] Total samples: {self.sample_count}")

        elif user == 'n':
            print("[Skipped]")

        elif user == 'o':
            self.current_label = "open"
            print("Label changed → open")

        elif user == 'c':
            self.current_label = "close_nothing"
            print("Label changed → close_nothing")

        elif user == 'g':
            self.current_label = "grabbed"
            print("Label changed → grabbed")

        else:
            print("Unknown input. Sample ignored.")

        # Done waiting
        self.waiting_for_user = False

    # ------------------------------------------------------------------
    # RECORDING CONTROL
    # ------------------------------------------------------------------
    def start_recording(self):
        self.recording = True
        self.sample_count = 0
        self.start_time = time.time()
        print("\n[Recording STARTED]\n")

    def stop_recording(self):
        self.recording = False
        print(f"\n[Recording STOPPED] {self.sample_count} samples saved.\n")

    def save_and_close(self):
        self.csv_file.close()
        print(f"\nSaved: {self.csv_path}")


# ----------------------------------------------------------------------
# MAIN LOOP — only active when NOT recording
# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    collector = ForceDataCollector()

    spin_thr = threading.Thread(target=rclpy.spin, args=(collector,), daemon=True)
    spin_thr.start()

    try:
        while True:
            if collector.recording:
                time.sleep(0.1)  # do NOT read input here
                continue

            cmd = input().strip().lower()

            if cmd == 'r':
                collector.start_recording()

            elif cmd == 's':
                collector.stop_recording()

            elif cmd == 'q':
                break

            else:
                print("Unknown command")
                collector.print_commands()

    except KeyboardInterrupt:
        pass

    collector.save_and_close()
    collector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
