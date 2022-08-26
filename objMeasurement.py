import cv2
import utils

print(cv2.__version__)

path = "C:/Users/kutka/Documents/python/Resources/objMeasure.jpg"

width = 1280
height = 720
bright = 200

scale = 3
wPaper = 208*scale
hPaper = 294*scale

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_BRIGHTNESS, bright)

webCam = True

while True:
    if webCam:
        success, frame = cam.read()
        img = frame.copy()
        frame = cv2.resize(frame,(0,0),None,0.5,0.5)
        #img = frame.copy()
    else:
        frame = cv2.imread(path)
        frame = cv2.resize(frame,(0,0),None,0.5,0.5)
        img = frame.copy()

    frame2, A4_Contours = utils.getContours(frame, showImg=False, drawContour=True, sidesNum=4)

    if len(A4_Contours) != 0:
        biggest = A4_Contours[0][2]
        #print(biggest)
        #print(A4_Contours[0][0])
        imgWarp = utils.warpImg(frame2, biggest, wPaper, hPaper)
        #cv2.imshow("A4 PAPER", imgWarp)

        frame3, object_Contours = utils.getContours(imgWarp, drawContour=False, sidesNum=4, minArea = 1500, color=(0,255,0), cThres=[50,50])
        #print(object_Contours[0][0])

        if len(A4_Contours) != 0:
            for obj in object_Contours:
                objPoints = utils.reorderPts(obj[2])
                #print(objPoints)
                objWidth = round((utils.findDisPythagoras(objPoints[0][0]//scale,objPoints[1][0]//scale))/10, 1) # / this returns float...// returns int
                objHeight = round((utils.findDisPythagoras(objPoints[0][0]//scale,objPoints[2][0]//scale))/10, 1)
                #print(objWidth)
                #print(objHeight)

                cv2.arrowedLine(frame3, (objPoints[0][0][0], objPoints[0][0][1]), (objPoints[1][0][0], objPoints[1][0][1]), (255, 0, 255), 3, 8, 0, 0.05)       
                cv2.arrowedLine(frame3, (objPoints[0][0][0], objPoints[0][0][1]), (objPoints[2][0][0], objPoints[2][0][1]), (255, 0, 255), 3, 8, 0, 0.05)
                                
                x, y, w, h = obj[3]
                cv2.putText(frame3, '{}cm'.format(objWidth), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)       
                cv2.putText(frame3, '{}cm'.format(objHeight), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)

            cv2.imshow("MY OBJECTS", frame3)

    cv2.imshow("MY FRAME", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

