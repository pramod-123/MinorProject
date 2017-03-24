import os
import cv2
faceCascade = cv2.CascadeClassifier('/home/pramod/PycharmProjects/MinorProject/haarcascades/haarcascade_frontalface_alt.xml')
fullbody=cv2.CascadeClassifier("/home/pramod/PycharmProjects/MinorProject/haarcascades/haarcascade_fullbody.xml")
file_path="/home/pramod/PycharmProjects/MinorProject/pics/"
video_capture = cv2.VideoCapture(0)
i=0
def take_pics(name):
    global i
    i=0
    new_file_path=None
    if os.path.exists(file_path):
        count=len(os.listdir(file_path))+1
        os.mkdir(file_path+str(count)+"."+name)
        new_file_path=file_path+str(count)+"."+name+"/"
    while True:

    # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)

    # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            crop_gray=gray[y:y+h,x:x+w]
            cv2.imshow('Video', crop_gray)
            resized_image=cv2.resize(crop_gray,(169,169))
            cv2.imwrite(new_file_path+str(i)+".jpg",resized_image)
            print i
            i+=1
        if i>100:
            cv2.destroyAllWindows()
            break


    # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture

if __name__ == '__main__':
    print "Take 100 pics to train the system"
    while True:
        name=raw_input("Enter the input ")
        take_pics(name)
        choise=raw_input("Want to add more people(y/n)")
        if choise!="y":
            break
video_capture.release()
cv2.destroyAllWindows()