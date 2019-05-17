# MMA8452Q i2c
import smbus

import threading
import time

from lib.state import State


class Acceleration:
    """
    The class acceleration handles the input from the MMA8452Q over i2c.

    Information is persisted to the state class.
    """

    class __impl:
        def __init__(self, logger):
            self.logger = logger
            self.bus = smbus.SMBus(1)
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.busReady = True
            self.state = State()

            self.thread = threading.Thread(target=self.measureThread)
            self.threadStop = False

            try:
                self.__initBus__()
            except Exception as e:
                self.busReady = False
                self.logger.logError(e)

        def __initBus__(self):
            # MMA8452Q address, 0x1d
            # Select Control register, 0x2A(42)
            #		0x00(00)	StandBy mode
            self.bus.write_byte_data(0x1d, 0x2A, 0x00)
            # MMA8452Q address, 0x1d
            # Select Control register, 0x2A(42)
            #		0x01(01)	Active mode
            self.bus.write_byte_data(0x1d, 0x2A, 0x01)
            # MMA8452Q address, 0x1d
            # Select Configuration register, 0x0E(14)
            #		0x00(00)	Set range to +/- 2g
            self.bus.write_byte_data(0x1d, 0x0E, 0x00)

            self.logger.debug("Acceleration module ready")

        def measureThreadStart(self):
            self.thread.start()

        def measureThreadStop(self):
            self.threadStop = True

        def measureThread(self):
            while (not self.threadStop):
                self.measure()
                time.sleep(self.state.ThreadSleepingThreshold)

        def measure(self):
            # MMA8452Q address, 0x1d
            # Read data back from 0x00(0), 7 bytes
            # Status register, X-Axis MSB, X-Axis LSB, Y-Axis MSB, Y-Axis LSB, Z-Axis MSB, Z-Axis LSB
            if (self.busReady):
                data = self.bus.read_i2c_block_data(0x1d, 0x00, 7)

                # Convert the data
                self.x = (data[1] * 256 + data[2]) / 16
                if self.x > 2047:
                    self.x -= 4096

                self.y = (data[3] * 256 + data[4]) / 16
                if self.y > 2047:
                    self.y -= 4096

                self.z = (data[5] * 256 + data[6]) / 16
                if self.z > 2047:
                    self.z -= 4096

                self.logger.measure(
                    "X: %d; Y: %d; Z: %d" % (self.x, self.y, self.z))

            self.state.x = self.x
            self.state.y = self.y
            self.state.z = self.z

            return [self.x, self.y, self.z]

    # SINGLETON EVERYTHING :D
    __inst = None

    def __init__(self, logger):
        # Check whether we already have an instance
        if Acceleration.__inst is None:
            Acceleration.__inst = Acceleration.__impl(logger)

        # Store instance reference in the handle
        self.__dict__["_Acceleration__inst"] = Acceleration.__inst

    # Delegate attribute getters/setters to instance
    def __getattr__(self, attr):
        return getattr(self.__inst, attr)

    def __setattr__(self, attr, value):
        return setattr(self.__inst, attr, value)
