import dlib
import cv2
import numpy as np

p = "./shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)

        for j in range(1, 68):
            cv2.putText(
                frame, 
                str(j), 
                (shape.part(j).x, shape.part(j).y), 
                fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                fontScale = 0.3,
                color=(0,0,255))
        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()