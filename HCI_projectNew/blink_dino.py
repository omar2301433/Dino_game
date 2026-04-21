from flask import Flask, render_template, Response
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import threading
import numpy as np

app = Flask(__name__)

cap = cv2.VideoCapture(0)
# Reduce camera resolution to lighten CPU load
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCount = 0
counter = 0
color = (255, 0, 255)
eyesClosed = False
lock = threading.Lock()

# ---------- Dino Game State (from blink_dino.py, adapted for OpenCV rendering) ----------
GAME_WIDTH, GAME_HEIGHT = 800, 400

dino_width = 40
dino_height = 60
dino_x = 50
dino_y = GAME_HEIGHT - dino_height - 30
dino_vel_y = 0
gravity = 1
jump = -15
is_jumping = False

cactus_width = 20
cactus_height = 50
cactus_x = GAME_WIDTH
cactus_y = GAME_HEIGHT - cactus_height - 30
cactus_vel = 7

ground_y = GAME_HEIGHT - 30
game_over = False


def update_game(blink_triggered: bool):
    """
    Update Dino game physics and handle jump on blink.
    Drawn later inside generate_frames.
    """
    global dino_y, dino_vel_y, is_jumping, cactus_x, game_over

    if game_over:
        # Simple auto-reset on next blink
        if blink_triggered:
            reset_game()
        return

    # Blink triggers jump
    if blink_triggered and not is_jumping:
        dino_vel_y = jump
        is_jumping = True

    # Dino physics
    dino_y += dino_vel_y
    dino_vel_y += gravity
    if dino_y >= GAME_HEIGHT - dino_height - 30:
        dino_y = GAME_HEIGHT - dino_height - 30
        dino_vel_y = 0
        is_jumping = False

    # Move cactus
    cactus_x -= cactus_vel
    if cactus_x < -cactus_width:
        cactus_x = GAME_WIDTH

    # Collision detection
    dino_rect = (dino_x, dino_y, dino_width, dino_height)
    cactus_rect = (cactus_x, cactus_y, cactus_width, cactus_height)

    if rects_collide(dino_rect, cactus_rect):
        game_over = True


def rects_collide(r1, r2) -> bool:
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (
        x1 + w1 < x2 or
        x1 > x2 + w2 or
        y1 + h1 < y2 or
        y1 > y2 + h2
    )


def reset_game():
    global dino_y, dino_vel_y, is_jumping, cactus_x, game_over
    dino_y = GAME_HEIGHT - dino_height - 30
    dino_vel_y = 0
    is_jumping = False
    cactus_x = GAME_WIDTH
    game_over = False


def generate_frames():
    global blinkCount, counter, color, ratioList, eyesClosed
    frame_index = 0  # used to skip FaceMesh on some frames for better FPS
    
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        blink_triggered = False

        # Only run heavy FaceMesh detection on every 2nd frame
        frame_index += 1
        run_detection = frame_index % 2 == 0
        faces = []

        if run_detection:
            img, faces = detector.findFaceMesh(img, draw=False)

            if faces:
                face = faces[0]

                # LEFT EYE
                leftUpper = face[159]
                leftLower = face[23]
                leftAtLeft = face[130]
                leftAtRight = face[243]

                lengthVerticalLeft, _ = detector.findDistance(leftUpper, leftLower)
                lengthHorizontalLeft, _ = detector.findDistance(leftAtLeft, leftAtRight)
                # Use float ratio for better sensitivity
                ratioLeft = (lengthVerticalLeft / lengthHorizontalLeft) * 100.0

                # RIGHT EYE
                rightUpper = face[386]
                rightLower = face[374]
                rightAtLeft = face[463]
                rightAtRight = face[362]

                lengthVerticalRight, _ = detector.findDistance(rightUpper, rightLower)
                lengthHorizontalRight, _ = detector.findDistance(rightAtLeft, rightAtRight)
                ratioRight = (lengthVerticalRight / lengthHorizontalRight) * 100.0

                # Show current eye ratios on the frame for easy tuning
                cv2.putText(
                    img,
                    f"L:{ratioLeft:.1f} R:{ratioRight:.1f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                ratioList.append((ratioLeft + ratioRight) / 2.0)
                if len(ratioList) > 3:
                    ratioList.pop(0)
                ratioAVG = sum(ratioList) / len(ratioList)

                # Tune this threshold based on the values you see on screen.
                # Typically open eyes are around 35–45 and closed eyes drop below ~30.
                bothEyesClosed = ratioLeft < 35 and ratioRight < 350

                if bothEyesClosed:
                    if not eyesClosed:
                        with lock:
                            blinkCount += 1
                        color = (0, 200, 0)
                        counter = 1
                        eyesClosed = True
                        blink_triggered = True  # new blink event -> jump
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

                # imgPlot = plotY.update(ratioAVG, color=color)
            else:
                # No face detected; just keep the frame as is
                ratioAVG = None

            # Resize camera frame to match game height
            frame_cam = cv2.resize(img, (GAME_WIDTH, GAME_HEIGHT))

            # ---------- Update and Render Dino Game ----------
            update_game(blink_triggered)

            # Create blank white canvas for the game
            game_img = np.ones((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8) * 255

            # Ground
            cv2.rectangle(game_img, (0, ground_y), (GAME_WIDTH, GAME_HEIGHT), (0, 200, 0), -1)

            # Dino
            cv2.rectangle(
                game_img,
                (dino_x, int(dino_y)),
                (dino_x + dino_width, int(dino_y + dino_height)),
                (0, 0, 0),
                -1,
            )

            # Cactus
            cv2.rectangle(
                game_img,
                (int(cactus_x), cactus_y),
                (int(cactus_x) + cactus_width, cactus_y + cactus_height),
                (0, 0, 0),
                -1,
            )

            # Game over text
            if game_over:
                cv2.putText(
                    game_img,
                    "Game Over - Blink to Restart",
                    (80, GAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

            # Combine camera and game side by side
            combined = cv2.hconcat([frame_cam, game_img])

            ret, buffer = cv2.imencode('.jpg', combined)
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
