import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'AttendanceImages'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print (classNames)



def grabEncodings(images=[]):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeimg)

    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        #print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')





encodeListKnownFaces = grabEncodings(images)
print(len(encodeListKnownFaces), "Encoding Complete")

cap = cv2.VideoCapture('SampleVideo/elonandjack.mp4')

while (cap.isOpened()):
    success , img = cap.read()
    # imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    facesCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame= face_recognition.face_encodings(imgS,facesCurrFrame)
    

    for encodeFace,faceLoc in zip(encodeCurrFrame,facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnownFaces,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnownFaces,encodeFace)

        #print(faceDis)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)

        y1,x2,y2,x1 = faceLoc
        # y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)

        markAttendance(name)

    cv2.imshow('Video',img)
    cv2.waitKey(0)

cap.release()