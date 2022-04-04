import cv2
import os
thres = 0.45
classNames= []
base_dir = os.path.dirname(__file__)
classFile = base_dir+'/coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
weights = base_dir+'/frozen_inference_graph.pb'
configuration = base_dir+'/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
net = cv2.dnn_DetectionModel(weights,configuration)
net.setInputSwapRB(True)
net.setInputMean((100, 100, 100))
net.setInputScale(1.0/ 100)
net.setInputSize(240,240)

from datetime import datetime
def func(img,folder_name):
    classIds, configs, box = net.detect(img,confThreshold=thres)
    if len(classIds) != 0:
        for classId, _,_ in zip(classIds.flatten(),configs.flatten(),box):
            if "cell phone"==classNames[classId-1]:    
                #print("Cell Phone detected")
                curr_time=datetime.now().strftime("%y_%m_%d_%H_%M_%S")
                cv2.imwrite("C:/Users/Nag/Documents/Proctoring_Images/"
    +folder_name+"/Phone_Detected_"+curr_time+".jpg",cv2.resize(img,(100,100)))