import cv2
import numpy as np
from skimage import feature
 

# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()    
    ret1, frame1 = cap1.read()
    print('frame1: ', frame1.shape)

    # Handles the mirroring of the current frame
    #frame = cv2.flip(frame,1)
    #frame1 = cv2.flip(frame1,1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    lbp = feature.local_binary_pattern(gray, 8, 2, method="nri_uniform")

    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    print('hist: ', hist)

    #combine = np.concatenate((frame,frame1),axis=1)
    # Saves image of the current frame in jpg file
    # name = 'frame' + str(currentFrame) + '.jpg'
    # cv2.imwrite(name, frame)

    # Display the resulting frame
    cv2.imshow('frame',frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()