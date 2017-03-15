import os
import numpy as np
import cv2
lables=[]
images = []
pics_paath="/home/pramod/PycharmProjects/MinorProject/pics/"
for i in os.listdir(pics_paath):
    if i.endswith('.png'):
        nmbr=int(i.split(".")[0])
        print nmbr
        pic=cv2.imread(pics_paath+i)
        images.append(pic)
        if nmbr>=77:
            lables.append(2)
        else:
            lables.append(1)
        cv2.imshow("pic",pic)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
model=cv2.createEigenFaceRecognizer()
print "training"
model.train(np.array(images),np.array(lables))
model.save("m.xml")
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
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        crop_gray = gray[y:y + h, x:x + w]
        resized_image = cv2.resize(crop_gray, (169, 169))
        lbl,conf=model.predict(resized_image)
    if lbl==2:
        print "shubham"
    elif lbl==1:
        print "Pramod"
    else:
        print "Unknown"
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()