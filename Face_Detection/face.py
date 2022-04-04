import playsound
from datetime import datetime
import cv2
import os
import numpy as np
base_dir = os.path.dirname(__file__)
cascPath = base_dir+"/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
 

def func(frame,folder_name):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60))
                                         
    if len(faces)!=1:
        curr_time=datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        sss="No_person_" if len(faces)==0 else "Multiple_People_Detected_"
        cv2.imwrite("C:/Users/Nag/Documents/Proctoring_Images/"
	+folder_name+"/"+sss+curr_time+".jpg",cv2.resize(image,(100,100)))