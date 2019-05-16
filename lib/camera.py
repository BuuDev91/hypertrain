import numpy as np

import os
import datetime
import time
import cv2
import sys
import imutils
import pytesseract
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
    upper_black_rgb = np.array([70, 70, 70])

    lower_blue_color = np.array([94, 80, 2])
    upper_blue_color = np.array([126, 255, 255])


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
            # self.tesseract = PyTessBaseAPI(psm=PSM.SINGLE_CHAR, path='/usr/share/tesseract-ocr/4.00/tessdata/')
            # self.tesseract.SetVariable("oem", "0")
            # self.tesseract.SetVariable("classify_bln_numeric_mode", "1")
            # self.tesseract.SetVariable("tessedit_char_whitelist", "123456789")
            self.filter = Filter()

            self.signalThresholdY = 150

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
            # oem 1 = LTSM, psm 10 = single char
            return pytesseract.image_to_string(pil_image, lang="eng", config="--oem 1 --psm 10")
            # self.tesseract.SetImage(pil_image)
            # return self.tesseract.GetUTF8Text()

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
            #bar = np.zeros((50, 300, 3), dtype="uint8")
            #startX = 0

            percent = 0.0
            color = None
            for (pcnt, clr) in zip(hist, centroids):
                if (percent < pcnt):
                    percent = pcnt
                    color = clr
                #endX = startX + (percent * 300)
                # cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                #              color.astype("uint8").tolist(), -1)
                #startX = endX

            # self.showImg("histogram", bar)

            rgb = color.astype("uint8").tolist()

            # self.logger.info("Cnt: " + str(self.cntNum) +
            #                 " " + str(rgb) + " " + str(percent)+"%")
            return rgb

        def analyzeRect(self, image, warped, box, x, y):
            #dominantColor = np.array(self.dominantColor(warped))
            # find amount of color blue in warped area, assuming over X% is the lap signal
            if (self.getAmountOfColor(warped, Colors.lower_blue_color, Colors.upper_blue_color, True) > 0.1):
                # if (cv2.inRange(dominantColor, np.array([80, 40, 20]), np.array([120, 60, 50]))):
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

            # elif (self.getAmountOfColor(warped, Colors.lower_white_color, Colors.upper_white_color) > 0.1):
            elif ((dominantColor >= 160).all() and (dominantColor <= 250).all()):
                color = "White"
            # self.logger.debug("Amount of White: " + str(self.getAmountOfColor(
            #    warped, Colors.lower_white_color, Colors.upper_white_color)))

            if (color):
                optimized = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                # cropValue: amount of the frame to be cropped out
                cropValue = 0
                optimized = optimized[cropValue:optimized.shape[0] -
                                      cropValue, cropValue: optimized.shape[1] - cropValue]
                optimized = cv2.resize(
                    optimized, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                # blur
                optimized = cv2.GaussianBlur(optimized, (5, 5), 0)

                # binary image
                optimized = cv2.threshold(
                    optimized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # binary inversion if dominant color is black
                if (color == "Black"):
                    optimized = cv2.bitwise_not(optimized)

                # now check the frame (2px) of the image.. there shouldn't be any noise since its a clean signal background
                h, w = optimized.shape[0:2]

                for iFrame in range(1, 2):
                    for iHeight in range(h):
                        if not(optimized[iHeight, iFrame]) or not(optimized[iHeight, w - iFrame]):
                            return False
                    for iWidth in range(w):
                        # or not(optimized[h - iFrame, iWidth])
                        if not(optimized[iFrame, iWidth]):
                            return False

                # cv2.imwrite("records/opt/" + str(self.cntNum) + ".jpg", optimized)

                result_txt = self.tesserOCR(optimized)

                # clean up output from tesseract
                result_txt = result_txt.replace("\n", "")
                result_txt = result_txt.replace(" ", "")

                if result_txt.isdigit() and 0 < int(result_txt) < 10:
                    self.showImg("opt " + str(self.cntNum), optimized)
                    if y <= self.signalThresholdY:
                        self.logger.info(
                            "Stop Signal OCR: " + result_txt + " X: " + str(x) + " Y: " + str(y))
                        self.state.setStopSignal(int(result_txt))
                        return "S: " + result_txt
                    elif self.state.StopSignalNum != 0:
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

            image_height = np.size(image, 0)
            image_width = np.size(image, 1)

            contourImage = image.copy()

            # focus only on the part of the image, where a signal could occur
            #image = image[int(image.shape[0] * 0.2):int(image.shape[0] * 0.8), 0:int(image.shape[1]*0.666)]

            mask = self.filter.autoCanny(image)

            # get a list of contours in the mask, chaining to just endpoints
            cnts = imutils.grab_contours(cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

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
                        if (y <= image_height * 0.2 or x >= image_width * 0.7):
                            continue

                        if (w <= 5 or h <= 5):
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

                        cv2.drawContours(contourImage, [box], 0, (255, 0, 0), 1)

                        if (h*2 < w and y <= self.signalThresholdY and rectContAreaRatio >= 80):
                            result = self.analyzeRect(
                                image, four_point_transform(image, [box][0]), box, x, y)
                            if (result):
                                cv2.drawContours(
                                    contourImage, [cnt], -1, (0, 255, 0), 2)
                                print(rectContAreaRatio)
                        # find all contours looking like a signal with minimum area
                        elif absoluteSizeToImageRatio >= 0.01:
                            result = None
                            # is it approx a square, or standing rect? then check for info or stop signal
                            if 0.2 <= sideRatio <= 1.1:
                                # transform ROI
                                if (sideRatio <= 0.9):
                                    coords, size, angle = rect
                                    size = size[0] + 8, size[1] + 4
                                    coords = coords[0] - 1, coords[1] + 0.9

                                    rect = coords, size, angle
                                    box = cv2.boxPoints(rect)
                                    box = np.int0(box)
                                
                                cv2.drawContours(contourImage, [box], 0, (0, 255, 0), 1)

                                warp = four_point_transform(image, [box][0])

                                result = self.analyzeSquare(
                                    image, warp, box, x, y)

                            if (result):
                                """ cv2.drawContours(
                                    contourImage, [cnt], -1, (0, 255, 0), 2) """
                                if (self.__imgOutput):
                                    cv2.drawContours(
                                        contourImage, [box], 0, (0, 0, 255), 1)
                                    M = cv2.moments(cnt)
                                    cX = int(M["m10"] / M["m00"])
                                    cY = int(M["m01"] / M["m00"])
                                    cv2.putText(contourImage, str(
                                        self.cntNum), (cX - 30, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
