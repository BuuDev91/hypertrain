# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import datetime
import cv2

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

from tesserocr import PyTessBaseAPI, PSM
from PIL import Image

# initialize the camera and grab a reference to the raw camera capture
#camera = PiCamera()
#camera.resolution = (640, 480)
#camera.framerate = 32
#camera.rotation = 180
#rawCapture = PiRGBArray(camera, size=(640, 480))
 
vs = PiVideoStream(resolution=(480, 360), framerate=32).start()
time.sleep(2)
fps = FPS().start()

# define range HSV for white color of the sign
sensitivity = 30
lower_color = np.array([0,0,255-sensitivity])
upper_color = np.array([255,sensitivity,255])

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(img):
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image
 
# capture frames from the camera
while True:
	image = vs.read()
	image = imutils.rotate(image, angle=180)

	frameArea = image.shape[0]*image.shape[1]
	
	# convert color image to HSV color scheme
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# define kernel for smoothing   
	kernel = np.ones((3,3),np.uint8)
	# extract binary image with active white regions
	mask = cv2.inRange(hsv, lower_color, upper_color)
	# morphological operations
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	cv2.imshow("mask", mask)
	
	# find contours in the mask
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

	# define variables to hold values during loop
	largestArea = 0
	largestRect = None
	largestAngle = 0
	sideRatio = 0
	
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		for cnt in cnts:
			# Rotated Rectangle. Here, bounding rectangle is drawn with minimum area,
			# so it considers the rotation also. The function used is cv2.minAreaRect().
			# It returns a Box2D structure which contains following detals -
			# ( center (x,y), (width, height), angle of rotation ).
			# But to draw this rectangle, we need 4 corners of the rectangle.
			# It is obtained by the function cv2.boxPoints()
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


				# find the largest rectangle within all contours
				if area > largestArea and sideRatio >= 0.8 and sideRatio <= 1.2:
					cv2.drawContours(image,[box],0,(0,255,0),2)
					largestArea = area
					largestRect = box
					largestAngle = angle
		
	if largestArea > frameArea*0.001:
		# draw contour of the found rectangle on  the original image   
		cv2.drawContours(image,[largestRect],0,(0,0,255),2)
		
		# cut and warp interesting area
		warped = four_point_transform(image, [largestRect][0])
		cv2.imshow("warped", warped)

		resized = imutils.resize(warped, height=500)
		gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
		smooth = remove_noise_and_smooth(gray)
		#crop image
		smooth = smooth[50:450, 50:450]
		cv2.imshow("smooth", smooth)

		result_txt = ""
		if True: #enable / disable ocr reading (slowing down)
			with PyTessBaseAPI(psm=10) as api:
				api.SetPageSegMode
				pil_image = Image.fromarray(smooth)
				api.SetImage(pil_image)
				result_txt = api.GetUTF8Text()
		
		#cv2.putText(image, "Angle: " + str(largestAngle) + " Ratio: " + str(sideRatio), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
		
		if result_txt:
			print(datetime.datetime.now().strftime("%H:%M:%S") + ": " + result_txt)
			cv2.putText(image, str(result_txt), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

	cv2.imshow("image", image)

	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break