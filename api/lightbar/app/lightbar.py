import logging
import threading
import time

import numpy as np

from .macros import LED0_ON_L, LED1_ON_L, LED2_ON_L, LED4_ON_L, LED5_ON_L, LED6_ON_L
from .zones import Zone

logger = logging.getLogger(__name__)

# Bus-recovery constants, named to match perihelion_cl/i2c_recovery.py.
SCL_PIN = 3
SCL_CYCLES = 9
SCL_CYCLE_DELAY = 1e-5


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
        # Serializes I2C writes vs. recovery operations. With aiohttp_retry's
        # tightened budget (start_timeout=1, per-request timeout=10) a stuck
        # write_i2c_block_data can be racing a retry on a different threadpool
        # thread; the lock keeps them from interleaving on the smbus2 handle.
        self._lock = threading.Lock()

    def step(self, action: np.ndarray):
        with self._lock:
            self.action = action.copy()
            action = self.ensure_safety_limits(action)
            self.safe_action = action
            duty_cycle = self.convert_to_duty_cycle(action)
            self.set_duty_cycle(duty_cycle)

    def reset(self):
        # PCA9685 software reset via I2C general call (datasheet section 7.6):
        # byte 0x06 sent to address 0x00 resets every PCA9685 on the bus to
        # power-up defaults. Mirrors legacy g2v_perihelion_v2.reset(). Note
        # that this leaves the chips in SLEEP mode until the next step() write.
        with self._lock:
            self.i2c.write_byte(0x00, 0x06)

    def scl_recover(self):
        # Bus-level recovery for a slave stuck mid-byte holding SDA low.
        # Closes /dev/i2c-1, steals SCL as a GPIO output, pulses it so the
        # stuck slave clocks out its remaining bits and releases SDA,
        # restores SCL to ALT0, and reopens /dev/i2c-1. Matches
        # perihelion_cl/i2c_recovery.py.
        import pigpio

        with self._lock:
            self.i2c.close()
            try:
                pi = pigpio.pi()
                try:
                    if not pi.connected:
                        raise RuntimeError("pigpiod is not running; cannot recover bus")
                    pi.set_mode(SCL_PIN, pigpio.OUTPUT)
                    pi.write(SCL_PIN, 1)
                    for _ in range(SCL_CYCLES):
                        time.sleep(SCL_CYCLE_DELAY)
                        pi.write(SCL_PIN, 0)
                        time.sleep(SCL_CYCLE_DELAY)
                        pi.write(SCL_PIN, 1)
                    pi.set_mode(SCL_PIN, pigpio.ALT0)
                finally:
                    pi.stop()
            finally:
                self.i2c = self.get_i2c()

    def set_duty_cycle(self, duty_cycles: np.ndarray):
        for channel in range(len(self.channels)):
            self.set_bar_pwm(channel, duty_cycles[:, channel])

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

    def set_bar_pwm(self, channel: int, duty_cycles: np.ndarray):
        for address, duty_cycle in zip(self.addresses, duty_cycles, strict=False):
            self.set_half_bar_pwm(address, channel, duty_cycle)

    def set_half_bar_pwm(self, address: int, channel: int, duty_cycle: int):
        """
        Set the PWM duty cycle for a channel on a half-bar.
        """
        command_array = self.get_command_array(channel, duty_cycle)
        try:
            self.i2c.write_i2c_block_data(address, 3, command_array)
        except OSError as e:
            logger.error(
                "i2c write failed: addr=0x%02x register=0x%02x duty=%d err=%s",
                address,
                command_array[1],
                duty_cycle,
                e,
            )
            raise
        time.sleep(0.05)

    def get_command_array(self, channel, duty_cycle):
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
