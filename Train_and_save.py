import os
import numpy as np
import cv2
lables=[]
prediction={}
prediction[-1]="unknown"
images = []
model=cv2.createLBPHFaceRecognizer()
model.load("m.yml")
pics_paath="/home/pramod/PycharmProjects/MinorProject/pics/"
for i in os.listdir(pics_paath):
    if not os.path.isdir(i):

        nmbr=int(i.split(".")[0])
        prediction[nmbr]=i.split(".")[1]
        for j in os.listdir(pics_paath+i+"/"):
            if True:
                pic = cv2.imread(pics_paath + i+"/"+j,cv2.IMREAD_GRAYSCALE)
                images.append(pic)
                lables.append(nmbr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print "training"
model.train(np.array(images),np.array(lables))
model.save("m.yml")
print "complete"



faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
i=0
cv2.destroyAllWindows()
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    lbl=-1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    conf=0.0
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        crop_gray = gray[y:y + h, x:x + w]
        resized_image = cv2.resize(crop_gray, (169, 169))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        lbl, conf = model.predict(resized_image)
        cv2.putText(frame,prediction[lbl],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()