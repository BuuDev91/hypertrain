import numpy as np

import datetime
import time
import cv2
import sys
import imutils

from lib.logger import Logger
from lib.state import State, Signal
from lib.filter import Compass

from imutils.perspective import four_point_transform
from imutils import contours
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image


class Colors:
    """
    Here are different color ranges defined for creating a bit mask of the image.

    """

    # define color boundaries
    lower_white_color = np.array([0, 15, 0])
    upper_white_color = np.array([180, 90, 255])

    lower_black_color = np.array([0, 0, 0])
    upper_black_color = np.array([180, 255, 60])

    lower_blue_color = np.array([90, 150, 60])
    upper_blue_color = np.array([150, 255, 255])


class Camera:
    """
    Class Camera is the eye of the train

    With OpenCV and imutils we capture and analyze the image from the videostream
    and try to detect signals like INFO, STOP or LAP signals and persist those image to the state class.
    """

    class __impl:

        def __init__(self, vs, imgOutput):
            self.__vs = vs
            self.__imgOutput = imgOutput
            self.logger = Logger()
            self.state = State()
            self.tesseract = PyTessBaseAPI(psm=PSM.SINGLE_CHAR)
            self.compass = Compass()

        def showImg(self, window, image):
            if self.__imgOutput:
                cv2.imshow(window, image)

        def warmup(self):
            time.sleep(1)
            self.tesserOCR(np.zeros((1, 1, 3), np.uint8))

        def tesserOCR(self, image):
            self.tesseract.SetVariable("classify_bln_numeric_mode", "1")
            pil_image = Image.fromarray(image)
            self.tesseract.SetImage(pil_image)
            return self.tesseract.GetUTF8Text()

        def analyzeArea(self, image, warped, box, x, y):
            color = ''
            # find amount of color black in warped area, assuming over X% is a numeric signal
            if (self.getAmountOfColor(warped, Colors.lower_black_color, Colors.upper_black_color) > 0.1):
                color = "Black"
                self.logger.debug("Amount of Black: " + str(self.getAmountOfColor(
                    warped, Colors.lower_black_color, Colors.upper_black_color)))

            # elif (self.getAmountOfColor(warped, Colors.lower_white_color, Colors.upper_white_color) > 0.3):
            #    color = "White"
            #    self.logger.debug("Amount of White: " + str(self.getAmountOfColor(
            #        warped, Colors.lower_white_color, Colors.upper_white_color)))

            if (color):
                # cropValue: amount of the frame to be cropped out
                cropValue = 6
                optimized = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                optimized = cv2.resize(
                    optimized, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
                optimized = optimized[cropValue:optimized.shape[1] -
                                      cropValue, cropValue:optimized.shape[0] - cropValue]
                optimized = cv2.GaussianBlur(optimized, (5, 5), 0)
                self.showImg("optimized", optimized)

                result_txt = ""
                if True:  # enable / disable ocr reading
                    result_txt = self.tesserOCR(optimized)

                result_txt = result_txt.replace("\n", "")
                result_txt = result_txt.replace(" ", "")
                if result_txt.isdigit() and int(result_txt) < 5 and int(result_txt) > 0:
                    cv2.putText(image, str(result_txt), (100, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                    if y <= 100:
                        self.logger.info(
                            "Stop Signal OCR: " + result_txt + " X: " + str(x) + " Y: " + str(y))
                        self.state.setStopSignal(int(result_txt))
                    else:
                        self.logger.info(
                            "Info Signal OCR: " + result_txt + " X: " + str(x) + " Y: " + str(y))
                        self.state.setCurrentSignal(
                            Signal.NUM, int(result_txt))

            # find amount of color blue in warped area, assuming over X% is the lap signal
            elif (self.getAmountOfColor(warped, Colors.lower_blue_color, Colors.upper_blue_color) > 0.1):
                #cv2.putText(image, "Rundensignal", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                self.logger.info("Rundensignal")
                self.state.setCurrentSignal(Signal.LAP)

        def getAmountOfColor(self, img, lowerColor, upperColor, convert2hsv=True):
            if (convert2hsv):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # create mask from color range
            maskColor = cv2.inRange(img, lowerColor, upperColor)
            # get ratio of active pixels
            ratio_color = cv2.countNonZero(maskColor) / (img.size)
            self.logger.debug("Ratio Color: " + str(ratio_color))
            return ratio_color

        # color picker for manual debugging color HSV range
        def pick_color(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pixel = hsv[y, x]
                color = np.array([pixel[0], pixel[1], pixel[2]])
                self.logger.info(color)

        # capture frames from the camera
        def capture(self):
            global hsv

            image = self.__vs.read()
            image = imutils.rotate(image, angle=0)

            # focus only on the
            # image = image[int(image.shape[0] * 0.2):int(image.shape[0]
            #                                            * 0.8), 0:int(image.shape[1]*0.666)]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            if (self.__imgOutput and False):  # hsv img output
                # convert color image to HSV color scheme
                cv2.namedWindow('hsv')
                cv2.setMouseCallback('hsv', self.pick_color)
                self.showImg('hsv', hsv)

            # maskBlack = cv2.inRange(
            #    image, np.array([0, 0, 0]), np.array([15, 15, 15]))
            mask = self.compass.greyEdgeDetector(image)
            #mask = cv2.bitwise_or(maskBlack, mask)
            #mask = cv2.bitwise_or(mask, maskWhite)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=2)

            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            #mask = cv2.Canny(mask, 35, 125)

            # get a list of contours in the mask, chaining to just endpoints
            _, cnts, _ = cv2.findContours(
                mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # loop contours
                for cnt in cnts:
                    rect = cv2.minAreaRect(cnt)
                    _, _, angle = rect
                    # approximate shape, if it has a length of 4 we assume its a rectangle
                    approx = cv2.approxPolyDP(
                        cnt, 0.04 * cv2.arcLength(cnt, True), True)

                    # the rectangle must not have a too big rotation (+/-10)
                    if len(approx) == 4 and (-90 <= angle <= -80 or angle >= -10):

                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(image, [box], 0, (0, 0, 255), 1)

                        (x, y, w, h) = cv2.boundingRect(approx)
                        sideRatio = w / float(h)

                        # calculate area of the rectangle
                        rArea = w * float(h)

                        # calculate area of the contour
                        cArea = cv2.contourArea(cnt)

                        areaRatio = rArea / cArea

                        # find all contours looking like a signal with minimum area
                        if 300 > rArea < 10000 and 0.8 <= areaRatio <= 1.2:  # and 0.8 <= sideRatio <= 1.2:
                            self.logger.debug("rectArea: " + str(rArea) + " contArea: " + str(
                                cArea) + " Angle: " + str(angle) + " SideRatio: " + str(sideRatio))
                            cv2.drawContours(image, [box], 0, (0, 255, 0), 1)
                            warped = four_point_transform(image, [box][0])
                            self.analyzeArea(image, warped, box, x, y)

            self.showImg("mask", mask)
            self.showImg("image", image)

    # Singleton
    __inst = None

    def __init__(self, vs=None, imageOutput=False):
        # Check whether we already have an instance
        if Camera.__inst is None:
            Camera.__inst = Camera.__impl(vs, imageOutput)

        # Store instance reference in the handle
        self.__dict__["_Camera__inst"] = Camera.__inst

    # Delegate attribute getters/setters to instance
    def __getattr__(self, attr):
        return getattr(self.__inst, attr)

    def __setattr__(self, attr, value):
        return setattr(self.__inst, attr, value)
