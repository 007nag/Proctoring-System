import playsound
from datetime import datetime
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
base_dir = os.path.dirname(__file__)
cascPath = base_dir+"/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
model = load_model(base_dir+"/mask_recog.h5") 
#video_capture = cv2.VideoCapture(0)
def func(frame,folder_name):
    #ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60))
                                         #flags=cv2.CASCADE_SCALE_IMAGE)
    faces_list=[]
    preds=[]
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h,x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame =  preprocess_input(face_frame)
        faces_list.append(face_frame)
    if len(faces_list)>0:
        preds = model.predict(faces_list)
        for pred in preds:
            (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        if mask<withoutMask:
            playsound.playsound("audio.mp3")
            curr_time=datetime.now().strftime("%y_%m_%d_%H_%M_%S")
            cv2.imwrite("C:/Users/Nag/Documents/Proctoring_Images/"
    +folder_name+"/No_Mask_"+curr_time+".jpg",cv2.resize(frame,(100,100)))