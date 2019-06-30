from datetime import datetime
import time
from enum import Enum


class Signal(Enum):
    NONE = 0
    LOWER = 1
    UPPER = 2
    LAP = 4


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
            self.StopSignalAnnounced = False
            self.StopSignalDistance = 0

            self.LastLapSignalTimeStamp = None
            self.LapSignalCount = 0
            self.LapSignalTimeThreshold = 3

            self.AccelerationPercent = 0
            self.LastAccelerationPercent = 0

            self.CoveredDistance = 0

            self.Loaded = False
            self.Approaching = Signal.NONE
            self.ApproachStop = False
            self.Stopped = True

            self.Standalone = False
            self.InvertCamera = False
            self.NoImageTransfer = False
            self.RecordImage = False
            self.MeasureMode = False

            self.ThreadSleepingThreshold = 0.3

            # acceleration parameters
            self.x = 0
            self.y = 0
            self.z = 0

        def reset(self):
            self.__init__()

        def setCurrentSignal(self, signal, num=0):
            self.CurrentSignal = signal
            self.CurrentNum = num
            self.CurrentSignalTimeStamp = time.time()

        def captureLapSignal(self):
            if (not self.LastLapSignalTimeStamp):
                self.LastLapSignalTimeStamp = time.time()

            # if seen lap signal within X seconds again, dont count it as a new lap
            if ((self.LastLapSignalTimeStamp + self.LapSignalTimeThreshold) <= time.time()):
                self.LastLapSignalTimeStamp = time.time()
                self.LapSignalCount += 1

        def setStopSignal(self, num):
            self.StopSignalNum = num
            self.StopSignalTimeStamp = time.time()
            self.StopSignalDistance = self.CoveredDistance

        def setStopSignalAnnounced(self, announced):
            self.StopSignalAnnounced = announced

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
