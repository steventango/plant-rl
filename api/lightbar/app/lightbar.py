import time

import numpy as np

from .macros import LED0_ON_L, LED1_ON_L, LED2_ON_L, LED4_ON_L, LED5_ON_L, LED6_ON_L
from .zones import Zone


class Lightbar:
    def __init__(self, zone: Zone):
        self.addresses = (zone.left, zone.right)
        self.channels = [
            "blue",
            "cool_white",
            "warm_white",
            "orange_red",
            "red",
            "far_red",
        ]
        self.i2c = self.get_i2c()
        self.action = None
        self.safe_action = None

    def step(self, action: np.ndarray):
        self.action = action.copy()
        action = self.ensure_safety_limits(action)
        self.safe_action = action
        duty_cycle = self.convert_to_duty_cycle(action)
        self.set_duty_cycle(duty_cycle)

    def set_duty_cycle(self, duty_cycles: np.ndarray):
        for channel in range(len(self.channels)):
            self.set_bar_pwn(channel, duty_cycles[:, channel])

    def ensure_safety_limits(self, action: np.ndarray):
        # safety limits
        # action should be 0.5 max per channel
        action /= 2
        # sum of all channels should be at most 2
        cond = action.sum(axis=1) > 2
        action[cond] /= action[cond].sum(axis=1, keepdims=True)
        action[cond] *= 2
        return action

    def convert_to_duty_cycle(self, action: np.ndarray):
        duty_cycle = action * 4095
        duty_cycle = duty_cycle.astype(np.int32)
        return duty_cycle

    def set_bar_pwn(self, channel: int, duty_cycles: np.ndarray):
        for address, duty_cycle in zip(self.addresses, duty_cycles, strict=False):
            self.set_half_bar_pwm(address, channel, duty_cycle)

    def set_half_bar_pwm(self, address: int, channel: int, duty_cycle: int):
        """
        Set the PWM duty cycle for a channel on a half-bar.
        """
        command_array = self.get_command_array(channel, duty_cycle)
        self.i2c.write_i2c_block_data(address, 3, command_array)
        time.sleep(0.025)

    def get_command_array(self, channel, duty_cycle):
        duty_cycle = max(0, duty_cycle)
        duty_cycle = min(4095, duty_cycle)
        duty_cycle = np.clip(duty_cycle, 0, 4095)
        low_byte = duty_cycle & 0xFF
        high_byte = duty_cycle >> 8
        if channel == 0:
            chip_register = LED0_ON_L
        elif channel == 1:
            chip_register = LED1_ON_L
        elif channel == 2:
            chip_register = LED6_ON_L
        elif channel == 3:
            chip_register = LED5_ON_L
        elif channel == 4:
            chip_register = LED4_ON_L
        elif channel == 5:
            chip_register = LED2_ON_L
        else:
            raise ValueError(f"perihelion; illegal channel number {channel}")
        command_array = [0, int(chip_register), 0, 0, int(low_byte), int(high_byte)]
        return command_array

    def get_i2c(self):
        from smbus2 import SMBus

        bus = SMBus(1)
        time.sleep(1)
        return bus
