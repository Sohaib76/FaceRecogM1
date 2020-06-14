import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file("ImageBasic/elon-musk.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)


imgTest = face_recognition.load_image_file("ImageBasic/elon.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(255,0,255),2)


faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0],faceLocTest[1],faceLocTest[2]),(255,0,255),2)
print(faceLocTest)
#1,2,3,0
#3,2,1,0
#0,1,2,3
#2.3.0,1
#3,0,1,2 approved
#3,1,0,2
#3,1,2,0
#3,0,2,1 approved
#3,2,0,1
#0,3,2,1
#1,3,2,0
#1,0,2,3
#2,0,1,3
#0,2,1,3

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results)
print(faceDis)


cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("elon Musk", imgElon)
cv2.imshow("elon Test", imgTest)
cv2.waitKey(0)