import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
prox_frame = frame



while True:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prox_frame_gray = cv2.cvtColor(prox_frame, cv2.COLOR_BGR2GRAY)
    
    frame_diff = cv2.absdiff(frame_gray, prox_frame_gray)
    
    ret, thresh = cv2.threshold(frame_diff, 127, 255, 0)
    
    
    imgGauss = cv2.GaussianBlur(frame_diff, (5,5),0)

    metodo = cv2.THRESH_BINARY_INV
    ret, imgBin = cv2.threshold(imgGauss, 100, 255, metodo)

    imgSeg = cv2.Canny(imgBin, 100, 200)

    
    (contours, hierarchy) = cv2.findContours(imgSeg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_x, max_x = 100000, 0
    min_y, max_y = 100000, 0
    for cout in contours:
        for coord in cout:
            x, y = coord[0]
            if(x > max_x):
                max_x = x
            if(x < min_x):
                min_x = x
            if(y > max_y):
                max_y = y
            if(y < min_y):
                min_y = y
        
    width = max_x - min_x
    height = max_y - min_y
    
    prox_frame = frame.copy()
    new_frame = frame.copy()
    
    if(min_x != 100000 and min_y != 100000):
        new_frame = cv2.rectangle(frame, (min_x, min_y),(min_x+width,min_y+height),(0,0,255),2)
            
    cv2.imshow('frame original ', new_frame)
    cv2.imshow('frame diff ',frame_diff)
    cv2.imshow('frame segmentado',imgSeg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()