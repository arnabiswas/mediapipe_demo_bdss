# Mediapipe demo
Eyelid and hand detection demos using webcam based on the mediapipe package.

## Installation
```bash
git clone git@github.com:arnabiswas/mediapipe_demo_bdss.git
cd mediapipe_demo_bdss
pip install -r requirements.txt
```

## Hand Detection
Detects hand landmarks and draws them on the webcam feed.
You can change the number of hands detected by changing the max_num_hands parameter in the mediapipe_hand.py file.
See [this](https://google.github.io/mediapipe/solutions/hands.html) for more details.

to run the demo:
```bash
python mediapipe_hand.py
```

## Eyelid Detection
Fits a mesh to your face and draws the eyelid landmarks of the detected face mesh on the webcam feed.
For more information see [this link](https://google.github.io/mediapipe/solutions/face_mesh)

to run the demo:
```bash
python mediapipe_eyelids.py
```