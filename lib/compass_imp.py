import numpy as np
import cv2


class Compass:
    #The prewitt kernels
    prewitt3 = np.array([[-1,1,1],[-1,-2,1],[-1,1,1]])
    prewitt4 = np.array([[1,1,1],[-1,-2,1],[-1,-1,1]])
    prewitt5 = np.array([[1,1,1],[1,-2,1],[-1,-1,-1]])
    prewitt6 = np.array([[1,1,1],[1,-2,-1],[1,-1,-1]])
    prewitt7 = np.array([[1,1,-1],[1,-2,-1],[1,1,-1]])
    prewitt8 = np.array([[1,-1,-1],[1,-2,-1],[1,1,1]])
    prewitt1 = np.array([[-1,-1,-1],[1,-2,1],[1,1,1]])
    prewitt2 = np.array([[-1,-1,1],[-1,-2,1],[1,1,1]])
    prewitt = [prewitt1, prewitt2, prewitt3, prewitt4, prewitt5, prewitt6, prewitt7, prewitt8]

    #The sobel kernels 
    sobel3 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel4 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    sobel5 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobel6 = np.array([[2,1,0],[1,0,-1],[0,-1,-2]])
    sobel7 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    sobel8 = np.array([[0,-1,-2],[1,0,-1],[2,1,0]])
    sobel1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel2 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
    sobel = [sobel1, sobel2,sobel3,sobel4,sobel5,sobel6,sobel7,sobel8]

    #The kirsch kernels
    kirsch3 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
    kirsch4 = np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
    kirsch5 = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
    kirsch6 = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
    kirsch7 = np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
    kirsch8 = np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
    kirsch1 = np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
    kirsch2 = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
    kirsch = [kirsch1, kirsch2,kirsch3,kirsch4,kirsch5,kirsch6,kirsch7,kirsch8]

    #The robinson kernels 
    robinson3 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    robinson4 = np.array([[0,1,1],[-1,0,1],[-1,-1,0]])
    robinson5 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    robinson6 = np.array([[1,1,0],[1,0,-1],[0,-1,-1]])
    robinson7 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    robinson8 = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    robinson1 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    robinson2 = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
    robinson = [robinson1, robinson2,robinson3,robinson4,robinson5,robinson6,robinson7,robinson8]

    def comapss(img, scale, offset, kernels=robinson):
        tmp_2d = img
        d_h, d_w = tmp_2d.shape
        mag = np.zeros((8, d_h, d_w))
        out_img = np.zeros(img.shape)
        
        max = 0
        #Apply a convolution
        for i, kernel in enumerate(kernels):
            mag[i] = np.abs(cv2.filter2D(tmp_2d, -1, kernel))

        #Scale and offset
        max = offset + mag.max(axis=0) * scale
        #Clip values
        max[max > 255] = 255
        max[max < 0] = 0

        return max.astype(np.uint8)
