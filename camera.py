# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import datetime
import cv2
import sys


from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
 
vs = PiVideoStream(resolution=(480, 360), framerate=32).start()
time.sleep(2)
fps = FPS().start()

# define range HSV for white color of the sign
sensitivity = 40
lower_white_color = np.array([0, 5, 200])
upper_white_color = np.array([180, 255, 255])

lower_black_color = np.array([0, 0, 0])
upper_black_color = np.array([180, 255, sensitivity])

lower_blue_color = np.array([100,225,100])
upper_blue_color = np.array([120,255,255])

def analyzeArea(image, warped):
		
	if (getAmountOfColor(warped, lower_blue_color, upper_blue_color) > 0.03): #tbd
		cv2.putText(image, "Rundensignal", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
		print(datetime.datetime.now().strftime("%H:%M:%S") + ": " + "Rundensignal")
	elif (getAmountOfColor(warped, lower_black_color, upper_black_color) > 0.1):
		#optimized = warped[50:450, 50:450]
		cropValue = 2
		optimized = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
		optimized = cv2.resize(optimized, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
		optimized = optimized[cropValue:optimized.shape[1] - cropValue, cropValue:optimized.shape[0] - cropValue]
		#optimized = cv2.GaussianBlur(optimized, (5, 5), 0)
		#optimized = cv2.adaptiveThreshold(optimized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
		
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
			print(datetime.datetime.now().strftime("%H:%M:%S") + ": " + result_txt)
			cv2.putText(image, str(result_txt), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

def getAmountOfColor(img, lowerColor, upperColor):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	maskColor = cv2.inRange(img, lowerColor, upperColor)
	ratio_blue = cv2.countNonZero(maskColor)/(img.size/3)

	#print("Ratio: " + str(ratio_blue))
	return ratio_blue

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv[y,x]

        upper =  np.array([pixel[0], pixel[1], pixel[2]])
        lower =  np.array([pixel[0], pixel[1], pixel[2]])
        print(lower, upper)

        image_mask = cv2.inRange(hsv,lower,upper)
        cv2.imshow("mask",image_mask)

global hsv, pixel
 
# capture frames from the camera
while True:

	image = vs.read()
	image = imutils.rotate(image, angle=180)

	frameArea = image.shape[0]*image.shape[1]
	
	#blurred = cv2.GaussianBlur(image, (11, 11), 0)
	# convert color image to HSV color scheme
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	cv2.namedWindow('hsv')
	cv2.setMouseCallback('hsv', pick_color)
	cv2.imshow("hsv", hsv)
	
	# extract binary image with defined color regions
	maskWhite = cv2.inRange(hsv, lower_white_color, upper_white_color)
	maskBlack = cv2.inRange(hsv, lower_black_color, upper_black_color)
	maskBlue = cv2.inRange(hsv, lower_blue_color, upper_blue_color)

	mask = cv2.bitwise_or(maskBlack, maskBlue)
	mask = cv2.bitwise_or(mask, maskWhite)

	#mask = cv2.erode(mask, None, iterations=2)
	#mask = cv2.dilate(mask, None, iterations=2)

	# define kernel for smoothing   
	kernel = np.ones((3,3),np.uint8)
	# morphological operations
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	cv2.imshow("mask", mask)
	
	# find contours in the mask
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		for cnt in cnts:
			rect = cv2.minAreaRect(cnt)
			x_y,l_w,angle = rect
			if angle and (angle >= -90 and angle <= -60 or angle >= -30):
				box = cv2.boxPoints(rect)
				box = np.int0(box)
			
				# count euclidian distance for each side of the rectangle
				sideOne = np.linalg.norm(box[0]-box[1])
				sideTwo = np.linalg.norm(box[0]-box[3])

				# count area of the rectangle
				area = sideOne*sideTwo
				sideRatio = sideOne / sideTwo

				#cv2.drawContours(image,[box],0,(0,255,0),2)

				# find all contours looking like a signal
				#if area > largestArea and sideRatio >= 0.8 and sideRatio <= 1.2:
				if area > 500 and sideRatio >= 0.8 and sideRatio <= 1.2:
					print("Area: " + str(area))
					cv2.drawContours(image,[box],0,(0,255,0),2)
					warped = four_point_transform(image, [box][0])
					analyzeArea(image, warped)

	cv2.imshow("image", image)

	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break