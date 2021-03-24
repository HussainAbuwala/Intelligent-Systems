# -*- coding: utf-8 -*-
"""
Created on Thu Oct 03 10:52:01 2019

This code is used from the link given by sir and modified for the assignment task
"""

import numpy as np
import cv2 as cv
import argparse
import time
import imutils
#parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                              The example file can be downloaded from: \
#                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
#parser.add_argument('image', type=str, help='path to image file')
#args = parser.parse_args()q
cap = cv.VideoCapture("single_final.mp4")
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
writer = None

try:
	prop = cv.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv.CAP_PROP_FRAME_COUNT
	total = int(cap.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1
    
#counter = 0
start = time.time()
history = []
while(1):
    ret,frame = cap.read()
    if(ret == False):
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if(p1 is None):
        index = len(history) - 1
        new = history[index][0]
        old = history[index][1]      
        a = old[0]
        b = old[1]
        c = new[0]
        d = new[1]
        p0 = np.asarray([[[c,d]]])
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        #cv.imshow('frame',frame_gray)    
        k = cv.waitKey(30) & 0xff
        if k == ord('q'): #quit when "q" is pressed
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        print(p0,p1)
       
        
    else:
    # Select good points (Crash when the person disappear because the tracking feature disappear as well. Can you solve this?)
        good_new = p1[st==1]
        good_old = p0[st==1]
        if(len(good_new) and len(good_old) > 0):
            history.append([(good_old[0][0],good_old[0][1]),
                            (good_new[0][0],good_new[0][1])])
        #history.append([good_old,good_new])
        
        
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
                  
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.arrowedLine(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
          
        img = cv.add(frame,mask)
        #cv.imshow('frame',img)    
        k = cv.waitKey(30) & 0xff
        if k == ord('q'): #quit when "q" is pressed
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    
    if writer is None:
		# initialize our video writer
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter("asgn_output/single_tracking/my_video_scenario-2/opticalflow.avi", fourcc, 30,(frame.shape[1], frame.shape[0]), True)

	# write the output frame to disk
    writer.write(frame)

end = time.time()
elap = end - start
print(total)
print(elap)
print(elap/total)

cap.release()
cv.destroyAllWindows()