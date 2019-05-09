from lib.camera import Camera
from lib.logger import Logger
from lib.communication import Communication
from lib.acceleration import Acceleration
from lib.state import State

import os
import cv2
import json
import time
import sys
import traceback

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS


def hyperloop():

    imageFolder = None
    imageNum = 1

    logger = Logger('relog')
    logger.setLogLevel('debug')
    logger.info('Started replay')

    state = State()
    for p in sys.argv:
        if (p != "" and os.path.isdir(p)):
            imageFolder = p

    camera = Camera(None, True)

    if imageFolder:
        # program loop
        while True:
            try:

                image = cv2.imread(os.path.join(
                    p, str(imageNum) + ".jpg"), 1)

                camera.capture(image)

                key = cv2.waitKey(0) & 0xFF

                if key == ord("n"):
                    if (os.path.exists(os.path.join(
                            p, str(imageNum + 1) + ".jpg"))):
                        imageNum += 1
                elif key == ord("b"):
                    if (os.path.exists(os.path.join(
                            p, str(imageNum - 1) + ".jpg"))):
                        imageNum -= 1
                elif key == ord('q'):
                    break

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.logError(e)
                traceback.print_exc(limit=3, file=sys.stdout)

    logger.info('Stopped')


if __name__ == '__main__':
    hyperloop()
