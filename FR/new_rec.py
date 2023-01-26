import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from timeit import default_timer as timer
from csv import writer

def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)

def makeRecord(elpased_time):
    timestamp = convert(elpased_time)
    with open('FR_Record.csv','a') as f:
        wobj = writer(f)
        wobj.writerow(["Match Found At: ", timestamp])
 
def encode_recognize():
    im_path = './FR/Images/target.jpg'
    img = cv2.imread(im_path)
    encodeList = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeList.append(encode)
        
    return encodeList
 
 

def cam_read(encodeListKnown):
    f = open("Record.csv", 'w+')
    f.close()
     
    vid_path = './FR/Videos/footage.mp4'
    cap = cv2.VideoCapture(vid_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('cv2_Output.mp4', 
                        cv2.VideoWriter_fourcc(*'MP4V'),
                        10, size)
    start = timer()
    while cap.isOpened():
        print('Video Streaming')
        success, img = cap.read()
        if not success:
            print('Video Stream Stopped')
            break 
        
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            #print(faceDis)
            matchIndex = np.argmin(faceDis)
    
            if matches[matchIndex]:
                name = 'Target'
                now = timer()
                makeRecord(now-start)
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
        
        result.write(img)
        
    
    print("Completed")
    return True
        
    