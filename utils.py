import cv2
import numpy as np
import mediapipe as mp

def getContours(img, cThres=[100,100], showImg=False, minArea=30000, sidesNum=0, drawContour=False, color=(0,0,255)):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, cThres[0], cThres[1])
    kernel = np.ones((5,5))
    imgDilate = cv2.dilate(imgCanny, kernel, iterations=3)
    imgErode = cv2.erode(imgDilate, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if showImg : cv2.imshow("ERODED IMAGE", imgErode)
    finalContours = []


    for i in contours:
        area = cv2.contourArea(i)
        if (area>=minArea):
            print(area)
            perimtr = cv2.arcLength(i, True)
            cornerPts = cv2.approxPolyDP(i, 0.02*perimtr, True)
            bbox = cv2.boundingRect(cornerPts)

            if sidesNum>0:
                if len(cornerPts) == sidesNum:
                    finalContours.append((len(cornerPts), area, cornerPts, bbox, i))
            else:
                finalContours.append((len(cornerPts), area, cornerPts, bbox, i))

    finalContours = sorted(finalContours, key = lambda x:x[1], reverse=True) # sorts contours based on their area(key = ...); reverse=True...sorts in descending order.

    if drawContour:
        for contour in finalContours:
            cv2.drawContours(img, contour[4], -1, color, 4) #contour[4] is i.

    return img, finalContours


def reorderPts(myPoints):
    #print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print(myPointsNew)
    return myPointsNew


def warpImg(img, points, w, h, buffer=20):
    #reorderPts(points)
    points = reorderPts(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    #print(imgWarp.shape)
    imgWarp = imgWarp[buffer:imgWarp.shape[0]-buffer,buffer:imgWarp.shape[1]-buffer] #we are using slicing here
    return imgWarp


def findDisPythagoras(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5
