import cv2
import numpy as np
import time


# cap = cv2.VideoCapture('./Videos/vid (1).mp4')
cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0
fps = 0

# def empty(x):
#   print(x)

"""
cv2.namedWindow("TrackBars")
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
"""

while True:
    pTime = time.time()
    ret, frame = cap.read()
    if ret:
        imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        """
        hmin = cv2.getTrackbarPos("Hue Min", "TrackBars")
        smin = cv2.getTrackbarPos("Sat Min", "TrackBars")
        vmin = cv2.getTrackbarPos("Val Min", "TrackBars")
        hmax = cv2.getTrackbarPos("Hue Max", "TrackBars")
        smax = cv2.getTrackbarPos("Sat Max", "TrackBars")
        vmax = cv2.getTrackbarPos("Val Max", "TrackBars")
        """
        # lower = np.array([9, 162, 0])
        # upper = np.array([20, 255, 255])
        lower = np.array([0, 0, 34])
        upper = np.array([179, 12, 255])
        #        lower = np.array([hmin, smin, vmin])
#        upper = np.array([hmax, smax, vmax])
#        print("hmin={},smin={}, vmin={},hmax={}, smax={}, vmax={}".format(hmin,smin,vmin,hmax,smax,vmax))
        mask = cv2.inRange(imgHSV, lower, upper)

        imgColor = cv2.bitwise_and(frame, frame, mask=mask)

        imgGray = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)

        ret, threshold = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)

        contours, hir = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            # cv2.drawContours(frame, cnt, -1, (0, 255, 255), 20)
            cv2.putText(frame, 'X:{},Y:{}'.format(x, y), (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 15)

        # cv2.imshow("img", frame)
        # cv2.imshow("mask", mask)

        if cv2.waitKey(300) & 0xFF == ord('q'):
            break
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        cv2.putText(frame, 'FPS:{}'.format(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 255), 2)
        cv2.imshow("img", frame)
cap.release()
cv2.destroyAllWindows()