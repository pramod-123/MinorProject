import cv2
model=cv2.createEigenFaceRecognizer()
print model.load("model.yml")