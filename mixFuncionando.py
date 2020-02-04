import numpy as np
import cv2
from collections import deque

from sklearn.externals import joblib
from skimage.feature import hog
import time


#------------------------------

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

#------------------------------

# Define the upper and lower boundaries for a color to be considered "Blue" rgb gbr
blueLower = np.array([110, 100, 100])#([100, 100, 100])#([100, 60, 60])
blueUpper = np.array([120, 255, 255])#([180, 255, 255])#([130, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Setup deques to store separate colors in separate arrays
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

bindex = 0
gindex = 0
rindex = 0
yindex = 0

colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Setup the Paint interface
paintWindow = np.zeros((300,300,3)) + 255

cv2.namedWindow('Paint',cv2.WINDOW_AUTOSIZE)

# Load the video
camera = cv2.VideoCapture(0)
camera.set(3,300)
camera.set(4,300)

# Keep looping
while True:
    # Grab the current paintWindow
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Add the coloring options to the frame
 


    # Check to see if we have reached the end of the video
    if not grabbed:
        break

    # Determine which pixels fall within the blue boundaries and then blur the binary image
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    
    #blueMask = cv2.erode(blueMask, kernel, iterations=2)
    
    #blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
   
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)
    cv2.imshow("mask", blueMask)
    # Find contours in the image
    (_,cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Check to see if any contours were found
    if len(cnts) > 0:
    	# Sort the contours and find the largest one -- we
    	# will assume this contour correspondes to the area of the bottle cap
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Get the moments to calculate the center of the contour (in this case Circle)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        bpoints[bindex].appendleft(center)
          
    # Append the next deque when no contours are detected (i.e., bottle cap reversed)
    else:
        bpoints.append(deque(maxlen=512))
        bindex += 1
      

    # Draw lines of all the colors (Blue, Green, Red and Yellow)
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 5)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 5)

    # Show the frame and the paintWindow image
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

	# If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
		
        break
#--------------------    
paintWindow=rotateImage(paintWindow,270)


#-----------------------
# Cleanup the camera and close any open windows
cv2.imwrite('test.jpg',paintWindow)
camera.release()
cv2.destroyAllWindows()

time.sleep(1)

# Load the classifier
clf = joblib.load("digits_cls3.pkl")

# Read the input image 
im = cv2.imread("test.jpg")
 #para contar el numero de pixeles negros

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #para contar el numero de pixeles negros
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
_,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (200, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    #cv2.extractChannel(roi,roi,1)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    nbr1 = clf.decision_function(np.array([roi_hog_fd], 'float64'))
    prob = clf.predict_proba(np.array([roi_hog_fd]))
   
    p=nbr1
    print('------------')
    tot =0
    """for nbs in nbr1:
		tot=tot+nbs
	print(tot)"""
	#print ndarray.sum(nbr1)
    print (prob)
    print('------------')
    print nbr1 [0,8]
    print np.amax(nbr1)
    print np.amin(nbr1)
    print('-------------')
    print('pixeles negros: '+ str(cv2.countNonZero(im_gray2)))
    numPixelBlancos=cv2.countNonZero(im_gray2)
    #print(clf.score(,,np.array([roi_hog_fd])))
    print prob[0,8]
    if prob[0,8]>0.8 and numPixelBlancos>87000 and numPixelBlancos<88750:
		print ('ok')
		cv2.putText(im, 'OK', (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    print('------------')
    print((nbr[0]))
    #grayPaint=cv2.cvtColor(paintWindow,cv2.COLOR_BGR2GRAY)
    
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()


