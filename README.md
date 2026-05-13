# Dino Game Controlled by Blink Detection

A web-based Dino Game controlled using eye blinks built with Python, Flask, OpenCV, MediaPipe Face Mesh, JavaScript, HTML, and CSS.  
The game allows players to control the dinosaur jump action using real-time blink detection through a webcam.

---

# Features

- Real-time blink detection
- Dino game controlled by eye blinks
- Webcam integration
- Face Mesh landmark tracking
- Responsive game interface
- Live browser gameplay
- Flask backend server
- Real-time communication between Python and frontend

---

# Tech Stack

## Frontend
- HTML
- CSS
- JavaScript

## Backend
- Python
- Flask

## Computer Vision
- OpenCV
- MediaPipe Face Mesh
- NumPy



# Libraries Used


Flask
opencv-python
mediapipe
numpy

## Installation

1. Clone the repository
git clone https://github.com/omar2301433/dino_game.git

3. Navigate to project folder
cd Dino_game

5. Create Virtual Environment (Optional)
##Windows

python -m venv venv
venv\Scripts\activate
or
pip install -r requirements.txt
How to Run
Start Flask Server
python app.py

## Open Browser

Visit:

http://127.0.0.1:5000
How the Game Works
Webcam captures live video
MediaPipe Face Mesh detects face landmarks
Eye blinking is detected using eye aspect ratio

## When a blink is detected:

The dinosaur jumps
Player avoids obstacles using blink-controlled jumps
Blink Detection Workflow
Detect face landmarks
Track eye points
Calculate blinking ratio
Trigger jump action when eyes close

## Modules

Backend
Webcam processing
Blink detection
Flask server handling
Frontend

## Dino game logic

Jump animations
Score tracking
Obstacle generation
Screenshots

Add screenshots for:

Dino game interface
Webcam blink detection
Face Mesh tracking
Gameplay screen
Future Improvements
Multiplayer mode
Score leaderboard
Sound effects
Difficulty levels
Mobile support
Head movement controls
