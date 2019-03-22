from camera import Camera
from logger import Logger
from communication import Communication
from acceleration import Acceleration
from state import State

import cv2
import time
import json

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS

vs = PiVideoStream(resolution=(480, 368), framerate=32).start()
camera = Camera(vs, True)

# camera warmup
time.sleep(1)

logger = Logger()
logger.setLogLevel('info')
logger.info('Started')

state = State()

communication = Communication(logger)
acceleration = Acceleration(logger)
# program loop
while True:

    # reads any input incoming over UART / i2c
    communication.read()

    # todo: set state from button press
    state.Stopped = False

    # todo: set state from arduino
    state.Loaded = True
    
    if (not state.Stopped and state.Loaded):
        # capture image from videostream and analyze 
        camera.capture()

        # measure acceleration
        acceleration.measure()

        data = {}
        data['sender'] = 'raspberry'
        data['action'] = 'accelerate'
        data['payload'] = 100

        # send message to arduino over UART
        #communication.write(json.dumps(data))

    # if the `q` key was pressed, break from the loop
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

logger.info('Stopped')
