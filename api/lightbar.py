import json
from pathlib import Path

import numpy as np

from api.macros import LED0_ON_L, LED1_ON_L, LED2_ON_L, LED4_ON_L, LED5_ON_L, LED6_ON_L


def convert_to_duty_cycle(action: np.ndarray):
    return (action * CALIBRATION_SCALE + CALIBRATION_MIN).astype(np.int32)


def set_duty_cycle(duty_cycle: np.ndarray):
    for address in ADDRESSES:
        for channel in range(NUM_CHANNELS):
            set_half_bar_pwm(address, channel, duty_cycle[address, channel])


def set_half_bar_pwm(bar_address: int, channel: int, duty_cycle: int):
    """
    Set the PWM duty cycle for a channel on a half-bar.
    """
    import smbus

    _i2c = smbus.SMBus(1)
    duty_cycle = max(0, duty_cycle)
    duty_cycle = min(4095, duty_cycle)
    low_byte = duty_cycle & 0xFF
    high_byte = duty_cycle >> 8
    if channel == 1:
        chip_register = LED0_ON_L
    elif channel == 2:
        chip_register = LED1_ON_L
    elif channel == 3:
        chip_register = LED4_ON_L
    elif channel == 4:
        chip_register = LED6_ON_L
    elif channel == 5:
        chip_register = LED5_ON_L
    elif channel == 6:
        chip_register = LED2_ON_L
    else:
        raise ValueError(f"perihelion; illegal channel number {channel}")

    command_array = [0, int(chip_register), 0, 0, int(low_byte), int(high_byte)]
    _i2c.write_i2c_block_data(bar_address, 3, command_array)
    time.sleep(0.025)


def load_calibration_data(addresses: list):
    with open(DIR_PATH / "lightbar_calibration.json", "r") as f:
        calibration_data = json.loads(f.read())
        calibration_data = {v["address"]: v for v in calibration_data}
        calibration_min = np.zeros((NUM_ADDRESSES, NUM_CHANNELS), dtype=np.int32)
        calibration_max = np.zeros((NUM_ADDRESSES, NUM_CHANNELS), dtype=np.int32)
        for i, address in enumerate(addresses):
            for channel, channel_name in enumerate(CHANNEL_NAMES):
                calibration_min[i, channel] = calibration_data[address][channel_name]["min"]
                calibration_max[i, channel] = calibration_data[address][channel_name]["max"]
        calibration_scale = calibration_max - calibration_min
    return calibration_min, calibration_scale


DIR_PATH = Path(__file__).parent

CHANNEL_NAMES = ["blue", "cool_white", "warm_white", "orange_red", "red", "far_red"]
NUM_CHANNELS = len(CHANNEL_NAMES)

with open(DIR_PATH / "addresses.json", "r") as f:
    ADDRESSES = json.loads(f.read())

NUM_ADDRESSES = len(ADDRESSES)


CALIBRATION_MIN, CALIBRATION_SCALE = load_calibration_data(ADDRESSES)
