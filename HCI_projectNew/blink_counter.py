import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot



cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCount = 0
counter = 0
color = (255, 0, 255)


while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img , draw=False)

    if faces:
        face = faces[0]
        #for id in idList:
            #cv2.circle(img, face[id], 5, (255, 0, 255), cv2.FILLED)

        leftUpper = face[159]
        leftLower = face[23] 
        leftAtLeft = face[130]
        leftAtRight = face[243]  
        
        lengthVertical, _ = detector.findDistance(leftUpper, leftLower)
        lengthHorizontal, _ = detector.findDistance(leftAtLeft, leftAtRight)

        #cv2.line(img, leftUpper, leftLower, (0, 200, 0), 3)
        #cv2.line(img, leftAtLeft, leftAtRight, (0, 200, 0), 3)

        print(int((lengthVertical/lengthHorizontal)*100))

        ratio = int((lengthVertical/lengthHorizontal)*100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAVG = sum(ratioList) / len(ratioList)
        
        if ratioAVG < 38 and counter == 0:
            blinkCount += 1
            color = (0, 200, 0)
            counter = 1

        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0  
                color = (255, 0, 255)
            
        cvzone.putTextRect(img, f'Blinks Count: {blinkCount}', (50, 100), colorR=color)


        imgPlot = plotY.update(ratioAVG, color=color)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
         img = cv2.resize(img, (840, 460))
         imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Live Camera", imgStack)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()