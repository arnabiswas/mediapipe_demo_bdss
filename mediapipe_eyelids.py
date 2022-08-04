import mediapipe as mp
import numpy as np
import cv2

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 

frame = cv2.imread('messi.jpg')
#cap = cv2.VideoCapture(0)
#ret,frame = cap.read()

img_h, img_w = frame.shape[:2]

with mp_face_mesh.FaceMesh(
	max_num_faces=1,
	refine_landmarks=False,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
) as face_mesh:

while True:
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	img_h, img_w = rgb_frame.shape[:2]
	results = face_mesh.process(rgb_frame)
	if results.multi_face_landmarks:
		mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
		cv2.polylines(frame, [mesh_points[LEFT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
		cv2.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
	cv2.imshow('img', frame)
	key = cv2.waitKey(1)
	if key ==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()