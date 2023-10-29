import cv2 as cv

trained_face_data = cv.CascadeClassifier('haarcascade-frontalface.xml')

# img  = cv.imread('photo.png')
# video = cv.VideoCapture('video.mp4')
# video.release()
webcam = cv.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()
    grayed_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_coorfinates = trained_face_data.detectMultiScale(grayed_img)

    # cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    # (x,y,w,h) = face_coorfinates[0]
    # cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

    for (x,y,w,h) in face_coorfinates:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)


    # print(face_coorfinates)

    cv.imshow('Pallav Face Detector App', frame)
    key = cv.waitKey(1)


    # Stop if Q is pressed
    if key == 81 or key ==113:
        break
print('Code Completed')