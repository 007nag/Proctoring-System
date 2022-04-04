import cv2 
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime 
import os
def on_trackbar(val):
    pass
last_time = datetime.now()
main_time=last_time.strftime("%y_%m_%d_%H_%M_%S")
#os.mkdir("saved_images/"+main_time)
cap= cv2.VideoCapture(0)
clas= cv2.CascadeClassifier("cascade_3_1.2_1.xml")
cv2.namedWindow('abc')
cv2.createTrackbar("scale", 'abc' , 11,100, on_trackbar)
cv2.createTrackbar("minne", 'abc' , 3,10, on_trackbar)
while True:
    min_ne=max(1,cv2.getTrackbarPos("minne", "abc"))
    scale=max(1.1,cv2.getTrackbarPos("scale", "abc")/10)
    print(scale,min_ne)#(1.1,2-3)
    _,fr= cap.read()
    fr= cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    #fr= cv2.resize(fr,(100,100))
    g= clas.detectMultiScale(fr,scale,min_ne)
    if len(g)>=1:
        if (datetime.now()-last_time).total_seconds()>=2:
            last_time= datetime.now()
            print('detected')
            #cv2.imwrite("saved_images/"+main_time+"/"+last_time.strftime("%y_%m_%d_%H_%M_%S")+".jpg",fr)
    for x,y,w,h in g:
        cv2.rectangle(fr,(x,y),(x+w,y+h),(255, 0, 0),1)
    #plt.imshow(fr,cmap=matplotlib.cm.gray)
    cv2.imshow('abc',fr)
    if cv2.waitKey(1)==27:break
cv2.destroyAllWindows()