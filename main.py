import os
from Face_Detection import face
from Mask_Detection import mask_detect
from Phone_Detection import phone1
import cv2
import numpy as np
from datetime import datetime 

cv2.namedWindow("Proctoring Software")
cv2.createTrackbar("Face?","Proctoring Software",1,1,lambda *a:None)
cv2.createTrackbar("Mask?","Proctoring Software",1,1,lambda *a:None)
cv2.createTrackbar("Phone?","Proctoring Software",1,1,lambda *a:None)
def main():
    folder_name= datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    os.mkdir("C:/Users/Nag/Documents/Proctoring_Images/"+folder_name)
    cap=cv2.VideoCapture(0)
    while True:
        _,fr=cap.read()
        if cv2.getTrackbarPos("Phone?","Proctoring Software"):
            phone1.func(fr,folder_name)    
        if cv2.getTrackbarPos("Mask?","Proctoring Software"):
            mask_detect.func(fr,folder_name)
        if cv2.getTrackbarPos("Face?","Proctoring Software"):
            face.func(fr,folder_name)
        if cv2.waitKey(1)==27:break
        cv2.imshow("Proctoring Software",cv2.resize(fr,(300,300)))
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()