import cv2
import logging as log
import matplotlib.pyplot as plt

video_capture = cv2.VideoCapture(0)

haar_face_cascade = cv2.CascadeClassifier('/home/student/PycharmProjects/face_detection/haarcascade_frontalface_default.xml')
haar_eye_cascade = cv2.CascadeClassifier('/home/student/PycharmProjects/face_detection/haarcascade_eye.xml')
haar_mouth_cascade = cv2.CascadeClassifier('/home/student/PycharmProjects/face_detection/Mouth.xml')

blue = (200, 100, 20)
red = (100, 55, 220)
green = (10, 255, 0)

def detect_faces(cascade, image, scaleFactor, minNeighbors, box_color, box_thickness):

    # convert the test image to gray image as opencv face detector expects gray images
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_img, scaleFactor, minNeighbors)

    # go over list of faces and draw them as rectangles on original colored
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), box_color, box_thickness)

    # log.info("Faces detected: " + str(len(faces)))

    return image

    # plt.imshow(convertToRGB(image))
    # plt.show()

    # print the number of faces found
    # print('Faces found:', len(faces))


# load cascade classifier training file for haarcascade

done = False

while not done:

    # captures each frame
    ret, frame = video_capture.read()

    face = detect_faces(haar_face_cascade, frame, 1.1, 2, blue, 3)
    eyes = detect_faces(haar_eye_cascade, frame, 1.3, 6, green, 2)
    mouth = detect_faces(haar_mouth_cascade, frame, 3.5, 15, red, 3)

    # display current live feed
    cv2.imshow("Face Detection", frame)

    # when "t" is pressed, terminate program
    if cv2.waitKey(1) & 0xFF == ord('t'):
        done = True