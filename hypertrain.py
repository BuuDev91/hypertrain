from camera import Camera
from logger import Logger
from communication import Communication
from acceleration import Acceleration

import cv2
import time

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS

vs = PiVideoStream(resolution=(480, 368), framerate=32).start()
camera = Camera(vs, False)

# camera warmup
time.sleep(1)

logger = Logger()
logger.setLogLevel('info')
logger.info('Started')

communication = Communication(logger)
acceleration = Acceleration(logger)
# program loop
while True:

    # reads any input incoming over UART
    communication.read()
    
    # capture image from videostream and analyze 
    camera.capture()

    # measure acceleration
    x, y, z = acceleration.measure()

    # send message to arduino over UART
    #communication.write("hello")

    # if the `q` key was pressed, break from the loop
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

logger.info('Stopped')
