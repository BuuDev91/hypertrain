# import the necessary packages
import numpy as np

import datetime
import cv2
import sys
import imutils

from logger import Logger

from imutils.perspective import four_point_transform
from imutils import contours
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image

logger = Logger()

# define color boundaries
lower_white_color = np.array([0, 15, 100])
upper_white_color = np.array([180, 255, 255])

lower_black_color = np.array([0, 0, 0])
upper_black_color = np.array([180, 255, 40])

lower_blue_color = np.array([100,200,30])
upper_blue_color = np.array([120,255,160])

# known dimensions of the sign for calibration purposes
KNOWN_WIDTH = 43
KNOWN_DISTANCE = 150
FOCAL_LENGTH = 600

global hsv, pixel

def analyzeArea(image, warped, box):
	
	# find amount of color black in warped area, assuming over X% is a numeric signal
	if (getAmountOfColor(warped, lower_black_color, upper_black_color) > 0.4):
		#print ("amount of black: " + str(getAmountOfColor(warped, lower_black_color, upper_black_color)))

		#PIXEL_WIDTH = box[0][1]

		# FOCAL_LENGTH must be calibrated before the distance to the known object can be determined
		#FOCAL_LENGTH = (PIXEL_WIDTH * KNOWN_DISTANCE) / KNOWN_WIDTH
		#print ("Focal length: " + str(FOCAL_LENGTH))

		#if (PIXEL_WIDTH > 0.0):
			#print ("Px: " + str(PIXEL_WIDTH))
			#DISTANCE = (KNOWN_WIDTH * FOCAL_LENGTH) / PIXEL_WIDTH
			#print ("Distance: " + str(DISTANCE))

		# cropValue: amount of the frame to be cropped out
		cropValue = 10
		optimized = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
		optimized = cv2.resize(optimized, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
		optimized = optimized[cropValue:optimized.shape[1] - cropValue, cropValue:optimized.shape[0] - cropValue]
		optimized = cv2.GaussianBlur(optimized, (5, 5), 0)
		cv2.imshow("optimized", optimized)

		result_txt = ""
		if True: #enable / disable ocr reading (slowing down)
			with PyTessBaseAPI(psm=PSM.SINGLE_CHAR) as api:
				api.SetVariable("classify_bln_numeric_mode", "1")
				pil_image = Image.fromarray(optimized)
				api.SetImage(pil_image)
				result_txt = api.GetUTF8Text()

		result_txt = result_txt.replace("\n", "")
		result_txt = result_txt.replace(" ", "")
		if result_txt.isdigit() and int(result_txt) < 5:
			logger.info("OCR: :" + result_txt)
			cv2.putText(image, str(result_txt), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

	# find amount of color blue in warped area, assuming over X% is the lap signal
	elif (getAmountOfColor(warped, lower_blue_color, upper_blue_color) > 0.1):
		#print ("amount of blue: " + str(getAmountOfColor(warped, lower_blue_color, upper_blue_color)))
		cv2.putText(image, "Rundensignal", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
		logger.info("Rundensignal")

def getAmountOfColor(img, lowerColor, upperColor, convert2hsv = True):
	if (convert2hsv):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	maskColor = cv2.inRange(img, lowerColor, upperColor)
	ratio_blue = cv2.countNonZero(maskColor)/(img.size/3)

	#print("Ratio: " + str(ratio_blue))
	return ratio_blue

#color picker for manual debugging color HSV range
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv[y,x]
        color =  np.array([pixel[0], pixel[1], pixel[2]])
        print(color)

 
# capture frames from the camera
def capture(vs):
	image = vs.read()
	image = imutils.rotate(image, angle=180)

	# convert color image to HSV color scheme
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	cv2.namedWindow('hsv')
	cv2.setMouseCallback('hsv', pick_color)
	cv2.imshow("hsv", hsv)
	
	# extract binary image with defined color regions
	maskWhite = cv2.inRange(hsv, lower_white_color, upper_white_color)
	maskBlack = cv2.inRange(hsv, lower_black_color, upper_black_color)
	maskBlue = cv2.inRange(hsv, lower_blue_color, upper_blue_color)

	# combine color masks bitwise or
	mask = cv2.bitwise_or(maskBlack, maskBlue)
	mask = cv2.bitwise_or(mask, maskWhite)

	# erode and dilate masks, smoothing
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# define kernel for smoothing   
	kernel = np.ones((3,3),np.uint8)
	# morphological operations
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	# leave just the outlines of the contours:
	# applying a gaussian blur and canny the outlines
	mask = cv2.GaussianBlur(mask, (5, 5), 0)
	mask = cv2.Canny(mask, 35, 125)

	cv2.imshow("mask", mask)
	
	# get a list of contours in the mask, chaining to just endpoints
	_,cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# loop contours
		for cnt in cnts:
			rect = cv2.minAreaRect(cnt)
			_,_,angle = rect
			# approximate shape, if it has a length of 4 we assume its a rectangle
			approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
			# the rectangle must not have a too big rotation (+/-30°)
			if len(approx) == 4 and angle and (angle >= -90 and angle <= -80 or angle >= -10):
				box = cv2.boxPoints(rect)
				box = np.int0(box)
			
				# count euclidian distance for each side of the rectangle
				sideOne = np.linalg.norm(box[0]-box[1])
				sideTwo = np.linalg.norm(box[0]-box[3])

				# calculate area of the rectangle
				area = sideOne*sideTwo
				sideRatio = sideOne / sideTwo

				# find all contours looking like a signal with minimum area
				if area > 750 and sideRatio >= 0.8 and sideRatio <= 1.2:
					logger.debug("Area: " + str(area) + " Angle: " + str(angle) + " SideRatio: " + str(sideRatio))
					cv2.drawContours(image,[box],0,(0,255,0),1)
					warped = four_point_transform(image, [box][0])
					analyzeArea(image, warped, box)

	cv2.imshow("image", image)