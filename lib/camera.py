import numpy as np
import numexpr as ne

import os
import datetime
import time
import cv2
import sys
import imutils

from lib.logger import Logger
from lib.state import State, Signal
from lib.filter import Filter

from sklearn.cluster import KMeans
from imutils.perspective import four_point_transform
from imutils import contours
from tesserocr import PyTessBaseAPI, PSM, OEM
from PIL import Image


class Colors:
    """
    Here are different color ranges defined for creating a bit mask of the image.

    """
    # define color boundaries
    lower_white_color = np.array([0, 15, 0])
    upper_white_color = np.array([180, 90, 255])

    lower_black_hsv = np.array([0, 0, 0])
    upper_black_hsv = np.array([180, 255, 60])

    lower_blue_color = np.array([90, 180, 100])
    upper_blue_color = np.array([130, 255, 255])


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
            self.image = None
            self.logger = Logger()
            self.state = State()
            self.tesseract = PyTessBaseAPI(
                psm=PSM.SINGLE_CHAR, oem=OEM.LSTM_ONLY, lang="digits")
            self.filter = Filter()

            self.signalThresholdY = 160
            self.LAPPatternSesibility = 5

            self.recordStamp = time.strftime(self.logger.timeFormat)
            self.recordNum = 0
            self.recordFolder = None
            self.cntNum = 0

            if (self.state.RecordImage):
                root = 'record'
                if not os.path.isdir(root):
                    os.mkdir(root)
                self.recordFolder = os.path.join(root, self.recordStamp)
                if not os.path.isdir(self.recordFolder):
                    os.mkdir(self.recordFolder)

        def showImg(self, window, image):
            if self.__imgOutput:
                cv2.imshow(window, image)

        def warmup(self):
            time.sleep(2.0)
            self.tesserOCR(np.zeros((1, 1, 3), np.uint8))

        def tesserOCR(self, image):
            self.tesseract.SetImage(Image.fromarray(image))
            return self.tesseract.GetUTF8Text(), self.tesseract.AllWordConfidences()

        def dominantColor(self, img, clusters=2):
            data = np.reshape(img, (-1, 3))
            data = np.float32(data)

            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, _, centers = cv2.kmeans(data, 1, None, criteria, 10, flags)
            return centers[0].astype(np.int32)

        def analyzeRect(self, image, warped, box, x, y):
            # find amount of color blue in warped area, assuming over X% is the lap signal
            if (self.getAmountOfColor(warped, Colors.lower_blue_color, Colors.upper_blue_color, True) > 0.1):
                self.logger.info("Rundensignal")
                self.state.setCurrentSignal(Signal.LAP)
                return "Rundensignal"

        def analyzeSquare(self, image, warped, box, x, y):

            #dominantColor, percent, _ = self.dominantColor(warped, 3)
            # dominantColor = self.dominantColor(
            #    cv2.cvtColor(warped, cv2.COLOR_BGR2HSV), 3)
            """  color = 'k'
             # find amount of color black in warped area, assuming over X% is a numeric signal
             if ((dominantColor <= 70).all()):
                 color = 'Black'

             elif ((dominantColor >= 180).all()):
                 color = 'White'

             if (color): """
            resizedWarp = cv2.resize(
                warped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            # gray
            optimized = cv2.cvtColor(resizedWarp, cv2.COLOR_BGR2GRAY)

            # blur
            optimized = cv2.GaussianBlur(optimized, (5, 5), 0)

            # binary image
            optimized = cv2.threshold(
                optimized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # binary inversion if dominant color is black
            """ if (color == 'Black'):
                optimized = cv2.bitwise_not(optimized) """

            # now check the frame (1px) of the image.. there shouldn't be any noise since its a clean signal background
            h, w = optimized.shape[0:2]
            clean = optimized[0, 0]
            for iFrame in range(0, 2):
                for iHeight in range(h):
                    if not(optimized[iHeight, iFrame] == clean) or not(optimized[iHeight, w - 1 - iFrame] == clean):
                        return False
                for iWidth in range(w):
                    # or not(optimized[h - iFrame, iWidth])
                    if not(optimized[iFrame, iWidth] == clean):
                        return False

            # cv2.imwrite("records/opt/" + str(self.cntNum) + ".jpg", optimized)

            output, confidence = self.tesserOCR(optimized)

            # if the resulting text is below X% confidence threshold, we skip it
            if not output or confidence[0] < 95:
                return False

            # clean up output from tesseract
            output = output.replace('\n', '')
            output = output.replace(' ', '')

            if output.isdigit() and 0 < int(output) < 10:
                """ self.showImg("opt " + str(self.cntNum),
                                np.hstack((resizedWarp, cv2.cvtColor(optimized, cv2.COLOR_GRAY2BGR)))) """
                if y <= self.signalThresholdY:
                    self.logger.info(
                        'Stop Signal OCR: ' + output + ' X: ' + str(x) + ' Y: ' + str(y) + ' Confidence: ' + str(confidence[0]) + '%')  # + ' DC: ' + str(dominantColor))
                    self.state.setStopSignal(int(output))
                    return 'S: ' + output
                elif self.state.StopSignalNum != 0:
                    self.logger.info(
                        'Info Signal OCR: ' + output + ' X: ' + str(x) + ' Y: ' + str(y) + ' Confidence: ' + str(confidence[0]) + '%')  # + ' DC: ' + str(dominantColor))
                    self.state.setCurrentSignal(
                        Signal.UPPER, int(output))
                    return 'I: ' + output

        def getAmountOfColor(self, img, lowerColor, upperColor, convert2hsv=True):
            if (convert2hsv):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # create mask from color range
            maskColor = cv2.inRange(img, lowerColor, upperColor)
            # get ratio of active pixels
            ratio_color = cv2.countNonZero(maskColor) / (img.size)
            return ratio_color

        # color picker for manual debugging
        def pick_color(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pixel = self.image[y, x]
                color = np.array([pixel[0], pixel[1], pixel[2]])
                self.logger.info(pixel)

        # capture frames from the camera
        def capture(self, savedImg=None):
            if (savedImg is not None):
                image = savedImg
            else:
                image = self.__vs.read()
                if (self.state.InvertCamera):
                    image = imutils.rotate(image, angle=180)

            self.image = image

            if (self.state.RecordImage):
                self.recordNum += 1
                cv2.imwrite(os.path.join(self.recordFolder,
                                         str(self.recordNum) + ".jpg"), image)
                return

            if (self.state.Approaching == Signal.UPPER or self.state.Approaching == Signal.LOWER):
                self.findNumberSignal(image)
            elif (self.state.Approaching == Signal.LAP):
                self.findLapSignal(image)

        def findLapSignal(self, image):
            contourImage = image.copy()

            blur = cv2.GaussianBlur(image, (3, 3), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            self.image = hsv
            mask = cv2.inRange(hsv, Colors.lower_blue_color,
                               Colors.upper_blue_color)

            cnts = imutils.grab_contours(cv2.findContours(
                mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE))

            if len(cnts) > 0:

                # transform all contours to rects
                rects = [cv2.boundingRect(cnt) for cnt in cnts]

                # now iterate all of the rects, trying to find an approximated sibiling shifted in Y-direction
                for rect in rects:
                    (x, y, w, h) = rect
                    cv2.rectangle(contourImage,(x,y),(x+w,y+h),(0,0,255),2)

                    # try to match the pattern from a given rect in all rects
                    counterPart = [counterRect for counterRect in rects if (
                        counterRect != rect and
                        x - 5 <= counterRect[0] <= x + 5 and
                        2*-(h+5) <= y - counterRect[1] <= 2*(h+5) and
                        w - 5 <= counterRect[2] <= w + 5) and
                        h - 5 <= counterRect[3] <= h + 5]

                    if (counterPart):
                        (x, y, w, h) = counterPart[0]
                        cv2.rectangle(contourImage,(x,y),(x+w,y+h),(0,255,0),2)
                        self.logger.info('LAP Signal')
                        self.state.captureLapSignal()
                        break

            self.showImg('contourImage', np.hstack(
                (hsv, contourImage, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))
            cv2.setMouseCallback('contourImage', self.pick_color)

        def findNumberSignal(self, image):

            image_height = np.size(image, 0)
            image_width = np.size(image, 1)

            contourImage = image.copy()

            # focus only on the part of the image, where a signal could occur
            # image = image[int(image.shape[0] * 0.2):int(image.shape[0] * 0.8), 0:int(image.shape[1]*0.666)]

            mask = self.filter.autoCanny(image, 2, 3)

            # get a list of contours in the mask, chaining to just endpoints
            cnts = imutils.grab_contours(cv2.findContours(
                mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE))

            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # loop contours
                for self.cntNum, cnt in enumerate(cnts):

                    rect = cv2.minAreaRect(cnt)
                    _, _, angle = rect

                    # approximate shape
                    approx = cv2.approxPolyDP(
                        cnt, 0.02 * cv2.arcLength(cnt, True), True)

                    # the rectangle must not have a too big rotation (+/-10)
                    # and more than 3 connecting points
                    if len(approx) >= 3 and (-90 <= angle <= -80 or angle >= -10):

                        box = cv2.boxPoints(rect)
                        box = np.int0(box)

                        (x, y, w, h) = cv2.boundingRect(approx)

                        # limit viewing range
                        if (y <= image_height * 0.2 or x >= image_width * 0.8):
                            continue

                        if (w <= 5 or h <= 5):
                            continue

                        # we are in approaching mode, thus we only care for the lower signals <= threshold
                        if ((self.state.Approaching == Signal.UPPER and y >= self.signalThresholdY) and not self.state.Standalone):
                            continue

                        sideRatio = w / float(h)

                        absoluteSizeToImageRatio = (
                            100 / (image_width * image_height)) * (w*h)

                        # calculate area of the bounding rectangle
                        rArea = w * float(h)

                        # calculate area of the contour
                        cArea = cv2.contourArea(cnt)
                        if (cArea):
                            rectContAreaRatio = (100 / rArea) * cArea
                        else:
                            continue

                        # cv2.drawContours(contourImage, [box], 0, (255, 0, 0), 1)
                        result = None

                        # is the rectangle sideways, check for lap signal
                        # if (h*2 < w and y <= self.signalThresholdY and rectContAreaRatio >= 80):
                        #result = self.analyzeRect(image, four_point_transform(image, [box][0]), box, x, y)
                        # find all contours looking like a signal with minimum area (1%)
                        if absoluteSizeToImageRatio >= 0.01:
                            # is it approx a square, or standing rect? then check for info or stop signal
                            if 0.2 <= sideRatio <= 1.1:
                                # transform ROI
                                if (sideRatio <= 0.9):
                                    coords, size, angle = rect
                                    size = size[0] + 8, size[1] + 4
                                    coords = coords[0] + 1, coords[1] + 1

                                    rect = coords, size, angle
                                    box = cv2.boxPoints(rect)
                                    box = np.int0(box)

                                """ cv2.drawContours(
                                    contourImage, [box], 0, (0, 255, 0), 1) """

                                warp = four_point_transform(image, [box][0])

                                result = self.analyzeSquare(
                                    image, warp, box, x, y)

                        if (result):
                            if (self.__imgOutput):
                                color = None
                                if (y >= self.signalThresholdY):
                                    color = (0, 0, 255)
                                else:
                                    color = (255, 0, 0)

                                cv2.drawContours(
                                    contourImage, [box], 0, color, 1)
                                cv2.drawContours(
                                    contourImage, [cnt], -1, color, 2)

                                """ M = cv2.moments(cnt)
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                cv2.putText(contourImage, str(
                                    self.cntNum), (cX - 30, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) """

                                self.logger.debug("[" + str(self.cntNum) + "] SideRatio: " + str(sideRatio) +
                                                  " AreaRatio: " + str(rectContAreaRatio) +
                                                  " ContArea: " + str(cArea) +
                                                  " RectArea: " + str(rArea) +
                                                  " AbsSize: " + str(absoluteSizeToImageRatio) +
                                                  " CntPoints: " + str(len(approx)) +
                                                  " Size: " + str(w) + "x" + str(h))

            """ if (self.__imgOutput):  # hsv img output
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                cv2.namedWindow('contourImage')
                cv2.setMouseCallback('contourImage', self.pick_color)
                # self.showImg("hsv", hsv) """

            self.showImg("contourImage", np.hstack(
                (contourImage, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))

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
