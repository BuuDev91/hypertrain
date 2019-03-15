import serial
               
class Communication:
    class __impl:
        def __init__(self, logger):
            self.logger = logger
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
            return incoming
        
        def write(self, message):
            if (message):
                self.logger.info("Sending: " + message)
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