import cv2
import imutils

haar_upper_body_cascade = cv2.CascadeClassifier("/home/paython/Desktop/haarcasecade/haarcascade_upperbody.xml")
haar_full_body_cascade = cv2.CascadeClassifier('/home/paython/Desktop/haarcasecade/haarcascade_fullbody.xml')
haar_lower_body_cascade = cv2.CascadeClassifier('/home/paython/Desktop/haarcasecade/haarcascade_lowerbody.xml')

# video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture("/home/paython/Videos/4K Video Downloader/Virtual mirror female digital signage.mp4")
video_width = video_capture.get(3)
video_height = video_capture.get(4)

while True:
    ret, frame = video_capture.read()

    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    upper_body = haar_upper_body_cascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 5,minSize = (50, 100),flags = cv2.CASCADE_SCALE_IMAGE)
    bodies = haar_full_body_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE)
    lower_body = haar_lower_body_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in upper_body:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
        cv2.putText(frame, "Upper Body Detected", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, "Body Detected", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)  # creates green color text with text size of 0.5 & thickness size of 2

    for (x, y, w, h) in lower_body:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame, "lower Body Detected", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Video', frame)


    if cv2.waitKey(1) ==13:
        break

video_capture.release()
cv2.destroyAllWindows()
