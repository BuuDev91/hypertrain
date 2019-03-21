import serial
import json

from enum import Enum

from state import State

# deserialized JSON Object
class JSONObject(object):
    def __init__(self, j):
        self.__dict__ = json.loads(j)

class Communication:
    class __impl:
        def __init__(self, logger):
            self.logger = logger
            self.state = State()
            self.serial = serial.Serial(            
                            port='/dev/serial0',
                            baudrate = 9600,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            bytesize=serial.EIGHTBITS,
                            timeout=0
                        )

        def read(self):
            incoming = ""
            if (self.serial.inWaiting() > 0):
                incoming = self.serial.read(self.serial.inWaiting()).decode('ascii')
                self.logger.info("Receiving: " + incoming)

                json = None
                try:
                    json = JSONObject(incoming)
                    if (json.sender == "arduino"):
                        if (json.action == "start"):
                            self.state.setStopped(False)
                        elif (json.action == "loaded"):
                            self.state.Loaded = True
                        elif (json.action == "stopped"):
                            self.state.setStopped(True)

                except AttributeError:
                    self.logger.error("AttributeError in JSON")
        
        def write(self, message):
            if (message):
                self.logger.debug("Sending: " + message)
                self.serial.write(message.encode())
    
    # Singleton 
    __inst = None
    def __init__(self, logger):
        # Check whether we already have an instance
        if Communication.__inst is None:
            Communication.__inst = Communication.__impl(logger)

        # Store instance reference in the handle
        self.__dict__["_Communication__inst"] = Communication.__inst

    # Delegate attribute getters/setters to instance
    def __getattr__(self, attr):
        return getattr(self.__inst, attr)

    def __setattr__(self, attr, value):
        return setattr(self.__inst, attr, value)