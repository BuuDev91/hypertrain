from lib.camera import Camera
from lib.logger import Logger
from lib.state import State, Signal

import os
import cv2
import json
import time
import sys
import traceback


import re


def sorted_aphanumeric(data):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def hyperloop():

    imageFolder = None
    imageNum = 0

    logger = Logger('relog')
    logger.setLogLevel('debug')
    logger.info('Started replay')

    state = State()
    for p in sys.argv:
        if (os.path.isdir(p)):
            imageFolder = p
        elif (p.isdigit()):
            imageNum = int(p)
        elif (p == "-lap"):
            state.Approaching = Signal.LAP
        elif (p == "-up"):
            state.Approaching = Signal.UPPER
        elif (p == "-lo"):
            state.Approaching = Signal.LOWER
        elif (p == "-s"):
            state.Approaching = Signal.UPPER

    if (state.Approaching != Signal.LAP):
        state.setStopSignal(1)

    camera = Camera(None, True)

    if imageFolder:
        # program loop
        files = sorted_aphanumeric(os.listdir(imageFolder))
        while True:
            try:
                file = os.path.join(imageFolder, files[imageNum])
                logger.info("["+str(imageNum)+"] Loaded file: " + file)
                image = cv2.imread(file, 1)

                camera.capture(image)

                key = cv2.waitKey(0) & 0xFF

                if key == ord("n"):
                    cv2.destroyAllWindows()
                    if (imageNum + 1 < len(files)):
                        imageNum += 1
                elif key == ord("b"):
                    cv2.destroyAllWindows()
                    if (imageNum - 1 >= 0):
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
