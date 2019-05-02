from lib.camera import Camera
from lib.logger import Logger
from lib.communication import Communication
from lib.acceleration import Acceleration
from lib.state import State

import cv2
import json
import time

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS

vs = PiVideoStream(resolution=(480, 368), framerate=32).start()
camera = Camera(vs, True)

# camera warmup
camera.warmup()

logger = Logger()
logger.setLogLevel('info')
logger.info('Started')

state = State()

communication = Communication(logger)
acceleration = Acceleration(logger)

# program loop
while True:
    try:
        # reads any input incoming over UART / i2c / GPIO
        communication.read()

        # todo: set state from button press
        #state.Stopped = False

        # todo: set state from arduino
        #state.Loaded = True

        if (not state.Stopped and state.Loaded):
            # capture image from videostream and analyze
            camera.capture()

            # measure acceleration
            acceleration.measure()

            if (state.StopSignalNum == 0):
                state.AccelerationPercent = 25
                communication.sendSpeedPercent(state.AccelerationPercent)
            # if we found our stop signal, announce it
            elif (state.StopSignalNum != 0 and not state.StopSignalAnnounced):
                communication.buzzSignalNumber(state.StopSignalNum)
                communication.sendSpeedPercent(100)
                state.setStopSignalAnnounced(True)
            # if we are close to passing round 2, we deccelerate to X percent
            elif (state.LapSignalCount >= 2 and not state.ApproachStop):
                communication.sendSpeedPercent(10)
                communication.sendApproachStop()
            elif (state.LapSignalCount < 2):
                state.AccelerationPercent = 100
                communication.sendSpeedPercent(state.AccelerationPercent)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    except KeyboardInterrupt:
        break
    except Exception as e:
        logger.error(str(e))

communication.sendStartStop('stop')
logger.info('Stopped')
