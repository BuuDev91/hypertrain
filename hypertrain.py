import camera
from logger import Logger

import cv2
import time

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS

vs = PiVideoStream(resolution=(480, 368), framerate=32).start()
#camera warmup
time.sleep(2)

logger = Logger()
logger.info("Started")

# program loop
while True:
    camera.capture(vs)

    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
