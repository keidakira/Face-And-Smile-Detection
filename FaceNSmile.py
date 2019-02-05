import cv2

smile_train_path = r'C:\Users\Nandan\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_smile.xml'
face_train_path = r'C:\Users\Nandan\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
smileCascade = cv2.CascadeClassifier(smile_train_path)
faceCascade = cv2.CascadeClassifier(face_train_path)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    smiles = smileCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=192,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
