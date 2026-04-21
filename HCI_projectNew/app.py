from flask import Flask, render_template, Response
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import threading

app = Flask(__name__)

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCount = 0
counter = 0
color = (255, 0, 255)
eyesClosed = False  
lock = threading.Lock()

def generate_frames():
    global blinkCount, counter, color, ratioList, eyesClosed
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        img, faces = detector.findFaceMesh(img , draw=False)

        if faces:
                face = faces[0]
                
                # LEFT EYE
                leftUpper = face[159]
                leftLower = face[23] 
                leftAtLeft = face[130]
                leftAtRight = face[243] 
                
                lengthVerticalLeft, _ = detector.findDistance(leftUpper, leftLower)
                lengthHorizontalLeft, _ = detector.findDistance(leftAtLeft, leftAtRight)
                ratioLeft = int((lengthVerticalLeft/lengthHorizontalLeft)*100)
                
                # RIGHT EYE
                rightUpper = face[386]
                rightLower = face[374]
                rightAtLeft = face[463]
                rightAtRight = face[362]
                
                lengthVerticalRight, _ = detector.findDistance(rightUpper, rightLower)
                lengthHorizontalRight, _ = detector.findDistance(rightAtLeft, rightAtRight)
                ratioRight = int((lengthVerticalRight/lengthHorizontalRight)*100)
                
                print(f"Left: {ratioLeft}, Right: {ratioRight}")
                
                
                ratioList.append((ratioLeft + ratioRight) / 2)
                if len(ratioList) > 3:
                    ratioList.pop(0)
                ratioAVG = sum(ratioList) / len(ratioList)
                
                
                bothEyesClosed = ratioLeft < 37 and ratioRight < 350
                
                if bothEyesClosed: 
                    if not eyesClosed:  
                        with lock:
                            blinkCount += 1
                        color = (0, 200, 0)
                        counter = 1
                        eyesClosed = True
                else:  
                    eyesClosed = False
                    color = (255, 0, 255)
                    counter = 0

                if counter != 0:
                    counter += 1
                    if counter > 10:
                        counter = 0  
                        color = (255, 0, 255)
                    
                cvzone.putTextRect(img, f'Blinks Count: {blinkCount}', (50, 100), colorR=color)

                #imgPlot = plotY.update(ratioAVG, color=color)
                img = cv2.resize(img, (640, 360))
                imgStack = cvzone.stackImages([img], 1, 1)
        else:
            img = cv2.resize(img, (840, 460))
            imgStack = cvzone.stackImages([img], 1, 1)

        ret, buffer = cv2.imencode('.jpg', imgStack)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset')
def reset():
    global blinkCount
    with lock:
        blinkCount = 0
    return {'status': 'reset'}

@app.route('/blink_count')
def get_blink_count():
    with lock:
        return {'count': blinkCount}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
