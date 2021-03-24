#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:16:52 2019
#This code supported by the tutorial in youtube
#https://www.youtube.com/watch?v=FbR9Xr0TVdY&t=445s
@author: hussainabuwala
"""

import numpy as np
import cv2
import imutils
import time



#==========================
# Read video from file
#==========================

#cap = cv2.VideoCapture('walk.avi')
#
#while(1):
#    ret, frame = cap.read()   
#    
#    if ret == True:
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        cv2.imshow('frame',gray)
#        if cv2.waitKey(30) & 0xff == ord('q'):
#            break
#    else:
#        break


#==========================
# Background subtraction
#==========================


cap = cv2.VideoCapture('single_final.mp4')
writer = None
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(cap.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1
    
start = time.time()
while(True):
    ret, frame = cap.read()#    
    if ret == True:
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(old_gray)
        
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
			#get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(c)
			
			#draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			
        #cv2.imshow('frame',fgmask)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break
    
    if writer is None:
		# initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("asgn_output/multiple_tracking/my_video_scenario-2/tt.avi", fourcc, 30,(frame.shape[1], frame.shape[0]), True)

	# write the output frame to disk
    writer.write(frame)
    


end = time.time()
elap = end - start
print(total)
print(elap)
print(elap/total)

# For further processing or release the memory
#==========================
#cv2.waitKey(0)
cap.release()
writer.release()
cv2.destroyAllWindows()



