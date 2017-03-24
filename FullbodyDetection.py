import os
import cv2

#faceCascade = cv2.CascadeClassifier('/home/pramod/PycharmProjects/MinorProject/haarcascades/haarcascade_frontalface_alt.xml')
fullbody=cv2.CascadeClassifier("/home/pramod/PycharmProjects/MinorProject/haarcascades/haarcascade_upperbody.xml")
#file_path="/home/pramod/PycharmProjects/MinorProject/pics/"
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body = fullbody.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in body:
        #crop_gray = gray[y:y + h, x:x + w]
        #resized_image = cv2.resize(crop_gray, (169, 169))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #lbl, conf = model.predict(resized_image)
        #cv2.putText(frame, prediction[lbl], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()