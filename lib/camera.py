import numpy as np

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
from tesserocr import PyTessBaseAPI, PSM
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

    lower_black_rgb = np.array([0, 0, 0])
    upper_blackc_rgb = np.array([70, 70, 70])

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
            self.tesseract.SetVariable("classify_bln_numeric_mode", "1")
            self.tesseract.SetVariable("tessedit_char_whitelist", "123456789")
            self.filter = Filter()

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
            time.sleep(1)
            self.tesserOCR(np.zeros((1, 1, 3), np.uint8))

        def tesserOCR(self, image):
            pil_image = Image.fromarray(image)
            self.tesseract.SetImage(pil_image)
            return self.tesseract.GetUTF8Text()

        def dominantColor(self, img, clusters=2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # reshaping to a list of pixels
            img = img.reshape((img.shape[0] * img.shape[1], 3))

            # using k-means to cluster pixels
            kmeans = KMeans(n_clusters=clusters)
            kmeans.fit(img)

            centroids = kmeans.cluster_centers_

            # create a histogram and label each section
            numLabels = np.arange(0, len(np.unique(kmeans.labels_)) + 1)
            (hist, _) = np.histogram(kmeans.labels_, bins=numLabels)

            # normalize the histogram, such that it sums to one
            hist = hist.astype("float")
            hist /= hist.sum()

            # plot
            bar = np.zeros((50, 300, 3), dtype="uint8")
            startX = 0

            percent = 0.0
            color = None
            for (pcnt, clr) in zip(hist, centroids):
                if (percent < pcnt):
                    percent = pcnt
                    color = clr
                endX = startX + (percent * 300)
                cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                              color.astype("uint8").tolist(), -1)
                startX = endX

            # self.showImg("histogram", bar)

            rgb = color.astype("uint8").tolist()

            self.logger.info("Cnt: " + str(self.cntNum) +
                             " " + str(rgb) + " " + str(percent)+"%")
            return rgb

        def analyzeRect(self, image, warped, box, x, y):
            # find amount of color blue in warped area, assuming over X% is the lap signal
            if (self.getAmountOfColor(warped, np.array([150, 50, 0]), np.array([255, 150, 100]), False) > 0.1):
                # cv2.putText(image, "Rundensignal", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                self.logger.info("Rundensignal")
                self.state.setCurrentSignal(Signal.LAP)
                return "Rundensignal"

        def analyzeSquare(self, image, warped, box, x, y):
            dominantColor = np.array(self.dominantColor(warped))
            color = ''
            # find amount of color black in warped area, assuming over X% is a numeric signal
            # if (self.getAmountOfColor(warped, Colors.lower_black_hsv, Colors.upper_black_hsv) > 0.1):
            if ((dominantColor <= 70).all()):
                color = "Black"
                # self.logger.debug("Amount of Black: " + str(self.getAmountOfColor(
                #    warped, Colors.lower_black_hsv, Colors.upper_black_hsv)))

            elif (self.getAmountOfColor(warped, Colors.lower_white_color, Colors.upper_white_color) > 0.1):
                color = "White"
                # self.logger.debug("Amount of White: " + str(self.getAmountOfColor(
                #    warped, Colors.lower_white_color, Colors.upper_white_color)))

            if (color):
                optimized = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                # cropValue: amount of the frame to be cropped out
                cropValue = 3
                optimized = optimized[cropValue:optimized.shape[1] -
                                      cropValue, cropValue:optimized.shape[0] - cropValue]
                optimized = cv2.resize(
                    optimized, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                # median blur
                optimized = cv2.blur(optimized, (5, 5))

                # binary image
                optimized = cv2.threshold(
                    optimized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # binary inversion if dominant color is black
                if (color == "Black"):
                    optimized = cv2.bitwise_not(optimized)

                self.showImg("opt " + str(self.cntNum), optimized)

                result_txt = self.tesserOCR(optimized)

                # clean up output form tesseract
                result_txt = result_txt.replace("\n", "")
                result_txt = result_txt.replace(" ", "")

                if result_txt.isdigit() and 0 < int(result_txt) < 10:
                    if y <= 150:
                        self.logger.info(
                            "Stop Signal OCR: " + result_txt + " X: " + str(x) + " Y: " + str(y))
                        self.state.setStopSignal(int(result_txt))
                        return "S: " + result_txt
                    else:
                        self.logger.info(
                            "Info Signal OCR: " + result_txt + " X: " + str(x) + " Y: " + str(y))
                        self.state.setCurrentSignal(
                            Signal.NUM, int(result_txt))
                        return "I: " + result_txt

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
                pixel = image[y, x]
                color = np.array([pixel[0], pixel[1], pixel[2]])
                self.logger.info(color)

        # capture frames from the camera
        def capture(self, savedImg=None):
            global image

            if (savedImg is not None):
                image = savedImg
            else:
                image = self.__vs.read()
                image = imutils.rotate(image, angle=0)

            contourImage = image.copy()

            # focus only on the part of the image, where a signal could occur
            # image = image[int(image.shape[0] * 0.2):int(image.shape[0]
            #                                            * 0.8), 0:int(image.shape[1]*0.666)]

            # maskBlack = cv2.inRange(
            #    image, np.array([0, 0, 0]), np.array([30, 30, 30]))
            blur = cv2.GaussianBlur(contourImage, (3, 3), 0)
            mask = self.filter.autoCanny(blur)

            #kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, None)
            #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # get a list of contours in the mask, chaining to just endpoints
            _, cnts, _ = cv2.findContours(
                mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # loop contours
                for self.cntNum, cnt in enumerate(cnts):

                    rect = cv2.minAreaRect(cnt)
                    _, _, angle = rect

                    # approximate shape, if it has a length of 4 we assume its a rectangle
                    approx = cv2.approxPolyDP(
                        cnt, 0.04 * cv2.arcLength(cnt, True), True)

                    # the rectangle must not have a too big rotation (+/-10)
                    if len(approx) == 4 and (-90 <= angle <= -80 or angle >= -10):

                        box = cv2.boxPoints(rect)
                        box = np.int0(box)

                        if (self.__imgOutput):
                            cv2.drawContours(
                                contourImage, [box], 0, (0, 0, 255), 1)

                        (x, y, w, h) = cv2.boundingRect(approx)
                        sideRatio = w / float(h)

                        # calculate area of the rectangle
                        rArea = w * float(h)

                        # calculate area of the contour
                        cArea = cv2.contourArea(cnt)
                        if (cArea):
                            areaRatio = rArea / cArea
                        else:
                            areaRatio = 0

                        # find all contours looking like a signal with minimum area
                        if 500 < rArea < 10000 and 0.7 <= areaRatio <= 1.3:
                            warp = four_point_transform(image, [box][0])

                            result = None
                            # is it approx a square? then check for info or stop signal
                            if 0.8 <= sideRatio <= 1.2:
                                if (self.__imgOutput):
                                    M = cv2.moments(cnt)
                                    cX = int(M["m10"] / M["m00"])
                                    cY = int(M["m01"] / M["m00"])
                                    cv2.putText(
                                        contourImage, str(self.cntNum), (cX - 30, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                                result = self.analyzeSquare(
                                    image, warp, box, x, y)
                            # if its a rectangle, this might be the lap signal
                            else:
                                result = self.analyzeRect(
                                    image, warp, box, x, y)

                            if (result):
                                cv2.drawContours(
                                    contourImage, [cnt], -1, (0, 255, 0), 2)

            self.showImg("mask", mask)

            if (self.__imgOutput and False):  # hsv img output
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                cv2.namedWindow('image')
                cv2.setMouseCallback('image', self.pick_color)
                # self.showImg("hsv", hsv)

            self.showImg("contourImage", contourImage)
            if (self.state.RecordImage):
                self.recordNum += 1
                cv2.imwrite(os.path.join(self.recordFolder,
                                         str(self.recordNum) + ".jpg"), image)

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
