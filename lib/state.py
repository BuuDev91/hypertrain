from datetime import datetime
from enum import Enum

class Signal(Enum):
    NONE = 0
    NUM = 1
    LAP = 2

class State:
    """
    Class State is used to persist train data.

    According to the state of the train, a next action will be evaulated.
    The state gets input from camera, arduino and sensors like button, acceleration module.
    """
    
    class __impl:
        def __init__(self):
            self.CurrentSignal = Signal.NONE
            self.CurrentNum = 0
            self.CurrentSignalTimeStamp = None
            self.StopSignalNum = 0
            self.StopSignalTimeStamp = None
            self.AccelerationPercent = 0
            self.Loaded = False
            self.ApproachStop = False
            self.Stopped = True
            # acceleration parameters
            self.x = 0
            self.y = 0
            self.z = 0
        
        def setCurrentSignal(self, signal, num = 0):
            self.CurrentSignal = signal
            self.CurrentNum = num
            self.CurrentSignalTimeStamp = datetime.now()

        def setStopSignal(self, num):
            self.StopSignalNum = num
            self.StopSignalTimeStamp = datetime.now()

        def setStopped(self, stopped):
            self.Stopped = stopped

    # Singleton, there can only be one state of the train
    __inst = None
    def __init__(self):
        # Check whether we already have an instance
        if State.__inst is None:
            State.__inst = State.__impl()

        # Store instance reference in the handle
        self.__dict__["_State__inst"] = State.__inst

    # Delegate attribute getters/setters to instance
    def __getattr__(self, attr):
        return getattr(self.__inst, attr)

    def __setattr__(self, attr, value):
        return setattr(self.__inst, attr, value)