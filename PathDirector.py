import cv2
faceCascade = cv2.CascadeClassifier('/home/pramod/PycharmProjects/MinorProject/haarcascades/haarcascade_frontalface_alt.xml')
fullbody=cv2.CascadeClassifier("/home/pramod/PycharmProjects/MinorProject/haarcascades/haarcascade_fullbody.xml")
video_capture = cv2.VideoCapture(0)



def find_the_grid(width_of_one_grid,x,y,w,h):
    left=int(x/width_of_one_grid)
    right=int((x+w)/width_of_one_grid)
    if right-left==2:
        return left+2
    if (right*width_of_one_grid)-x > (x+w)-(right*width_of_one_grid):
        return left+1
    else:
        return right+1


    return
devide_in=5
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
        frame_width=frame.shape[1]
        frame_height=frame.shape[0]
        crop_gray = gray[y:y + h, x:x + w]
        resized_image = cv2.resize(crop_gray, (169, 169))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(find_the_grid(int(frame_width/devide_in),x,y,w,h)), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.line(frame, (frame_width / 5,0), (frame_width/5,frame_height ), (0, 255, 0), thickness=3, lineType=8,
                 shift=0)
        cv2.line(frame, (2*frame_width / 5, 0), (2*frame_width / 5, frame_height), (0, 255, 0), thickness=3, lineType=8,
                 shift=0)
        cv2.line(frame, (3*frame_width / 5, 0), (3*frame_width / 5, frame_height), (0, 255, 0), thickness=3, lineType=8,
                 shift=0)
        cv2.line(frame, (4*frame_width / 5, 0), (4*frame_width / 5, frame_height), (0, 255, 0), thickness=3, lineType=8,
                 shift=0)
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
