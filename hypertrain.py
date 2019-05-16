from lib.camera import Camera
from lib.logger import Logger
from lib.communication import Communication
from lib.acceleration import Acceleration
from lib.state import State

import cv2
import json
import time
import sys
import traceback

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
import picamera


def hyperloop():

    logger = Logger()
    logger.setLogLevel('info')
    logger.info('Started')

    state = State()
    for p in sys.argv:
        if (p == "--standalone" or p == "-s"):
            state.Standalone = True
            logger.info("Standalone mode activated")
        elif (p == "--nocamera" or p == "-nc"):
            state.NoImageTransfer = True
            logger.info("Camera image transfer X11 disabled")
        elif (p == "--record" or p == "-r"):
            state.RecordImage = True
            logger.info("Record mode activated")

    vs = PiVideoStream(resolution=(480, 368), framerate=32).start()
    piCamera = vs.camera
    piCamera.exposure_mode = 'sports'
    piCamera.ISO = 1600
    camera = Camera(vs, not state.NoImageTransfer)

    # camera warmup
    camera.warmup()

    communication = Communication(logger)
    acceleration = Acceleration(logger)

    # program loop
    while True:
        try:
            # reads any input incoming over UART / i2c / GPIO
            communication.read()

            # measure acceleration
            acceleration.measure()

            if ((not state.Stopped and state.Loaded) or state.Standalone):
                # capture image from videostream and analyze
                camera.capture()

                if (state.StopSignalNum == 0):
                    state.AccelerationPercent = 25
                    communication.sendSpeedPercent(state.AccelerationPercent)
                # if we found our stop signal, announce it
                elif (state.StopSignalNum != 0 and not state.StopSignalAnnounced):
                    communication.buzzSignalNumber(state.StopSignalNum)
                    communication.sendSpeedPercent(100)
                    state.setStopSignalAnnounced(True)
                # if we are close to passing round 2, we deccelerate to X percent
                elif (state.LapSignalCount >= 3 and not state.ApproachStop):
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
            logger.logError(e)
            traceback.print_exc(limit=3, file=sys.stdout)

    communication.sendStartStop('stop')
    logger.info('Stopped')


if __name__ == '__main__':
    hyperloop()
