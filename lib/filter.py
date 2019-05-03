import numpy as np
import cv2


class Filter:
    class __impl:
        def __init__(self):
            # prewitt kernels
            prewitt1 = np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]])
            prewitt2 = np.array([[-1, -1, 1], [-1, -2, 1], [1, 1, 1]])
            prewitt3 = np.array([[-1, 1, 1], [-1, -2, 1], [-1, 1, 1]])
            prewitt4 = np.array([[1, 1, 1], [-1, -2, 1], [-1, -1, 1]])
            prewitt5 = np.array([[1, 1, 1], [1, -2, 1], [-1, -1, -1]])
            prewitt6 = np.array([[1, 1, 1], [1, -2, -1], [1, -1, -1]])
            prewitt7 = np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]])
            prewitt8 = np.array([[1, -1, -1], [1, -2, -1], [1, 1, 1]])
            self.prewitt = [prewitt1, prewitt2, prewitt3,
                            prewitt4, prewitt5, prewitt6, prewitt7, prewitt8]

            # sobel kernels
            sobel1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            sobel2 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
            sobel3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel4 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
            sobel5 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            sobel6 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
            sobel7 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            sobel8 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
            self.sobel = [sobel1, sobel2, sobel3,
                          sobel4, sobel5, sobel6, sobel7, sobel8]

            # kirsch kernels
            kirsch1 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
            kirsch2 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
            kirsch3 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
            kirsch4 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
            kirsch5 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
            kirsch6 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
            kirsch7 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
            kirsch8 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
            self.kirsch = [kirsch1, kirsch2, kirsch3,
                           kirsch4, kirsch5, kirsch6, kirsch7, kirsch8]

            # robinson kernels
            robinson1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            robinson2 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
            robinson3 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            robinson4 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
            robinson5 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            robinson6 = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])
            robinson7 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            robinson8 = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
            self.robinson = [robinson1, robinson2, robinson3,
                             robinson4, robinson5, robinson6, robinson7, robinson8]

        def colorEdgeDetector(self, img):
            tmp_2d = img
            d_h, d_w, d = tmp_2d.shape
            mag = np.zeros((8, d_h, d_w), dtype=np.int16)

            newImage = np.zeros((d_h, d_w), dtype=np.int16)
            for channel in range(d):
                for i, kernel in enumerate(self.prewitt):
                    mag[i] = np.abs(cv2.filter2D(
                        tmp_2d[:, :, channel], cv2.CV_16S, kernel))

            return mag.max(axis=0).astype(np.uint8)

        def greyEdgeDetector(self, imgcolor):
            imggrey = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)
            img = cv2.threshold(imggrey, 128, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            d_h, d_w = img.shape
            mag = np.zeros((8, d_h, d_w))
            out_img = np.zeros(img.shape)

            for i, kernel in enumerate(self.prewitt):
                mag[i] = np.abs(cv2.filter2D(img, -1, kernel))

            return mag.max(axis=0).astype(np.uint8)

        def autoCanny(self, img, sigma=0.33):

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = cv2.GaussianBlur(gray, (3, 3), 2)

            # compute the median of the single channel pixel intensities
            v = np.median(image)

            # apply automatic Canny edge detection using the computed median
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edged = cv2.Canny(image, lower, upper)

            # return the edged image
            return edged

    # Singleton
    __inst = None

    def __init__(self):
        # Check whether we already have an instance
        if Filter.__inst is None:
            Filter.__inst = Filter.__impl()

        # Store instance reference in the handle
        self.__dict__["_Filter__inst"] = Filter.__inst

    # Delegate attribute getters/setters to instance
    def __getattr__(self, attr):
        return getattr(self.__inst, attr)

    def __setattr__(self, attr, value):
        return setattr(self.__inst, attr, value)
