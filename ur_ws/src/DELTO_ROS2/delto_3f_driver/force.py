import time
from pymodbus.client.tcp import ModbusTcpClient

# Gripper Connection Details
GRIPPER_IP = "169.254.186.72"  # Change if necessary
PORT = 502  # Try 10000 if 502 does not work
SLAVE_ID = 1  # Modbus slave ID
CURRENT_REG_START = 0x0E  # Start address for motor current registers
NUM_MOTORS = 12  # Total number of motors
READ_INTERVAL = 0.5  # Seconds between readings (adjust as needed)

def convert_to_signed(value):
    """ Convert unsigned 16-bit integer to signed 16-bit integer """
    return value if value < 32768 else value - 65536

def estimate_force(motor_current):
    """ Convert motor current (mA) to estimated force (N) using a scaling factor """
    k = 0.01  # Adjust this based on calibration
    return round(k * motor_current, 2)

def read_motor_current(client):
    """ Reads only 4th, 8th, and 12th motor current values """
    try:
        # Read all 12 motor current registers
        response = client.read_input_registers(address=CURRENT_REG_START, count=NUM_MOTORS, slave=SLAVE_ID)
        if response.isError():
            print("❌ Modbus Read Error!")
            return None

        raw_current = response.registers

        # Select only 4th (index 3), 8th (index 7), and 12th (index 11)
        selected_currents = [convert_to_signed(raw_current[i]) for i in [3, 7, 11]]
        estimated_forces = [estimate_force(c) for c in selected_currents]

        return selected_currents, estimated_forces

    except Exception as e:
        print(f"❌ Error reading Modbus registers: {e}")
        return None

if __name__ == "__main__":
    client = ModbusTcpClient(GRIPPER_IP, port=PORT, timeout=3)
    if not client.connect():
        print(f"❌ Failed to connect to ({GRIPPER_IP}, {PORT})")
        exit(1)

    try:
        while True:
            motor_current, estimated_force = read_motor_current(client)
            if motor_current is not None:
                print(f"Selected Motor Currents (mA) [4th, 8th, 12th]: {motor_current}")
                print(f"Estimated Forces (N) [4th, 8th, 12th]: {estimated_force}\n")
            time.sleep(READ_INTERVAL)  # Adjust delay for desired frequency

    except KeyboardInterrupt:
        print("🔴 Stopping force reading...")
    finally:
        client.close()

