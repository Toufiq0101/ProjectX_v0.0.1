import cv2
import os

new_path = 'D:/Project_v0.0.0/'

def save(img, name, bbox, width=180, height=227):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    imgCrop = cv2.resize(imgCrop, (width, height))
    cv2.imwrite(name + ".jpg", imgCrop)

def face_detection_in_video(video_path):
    # Open the video capture
    cap = cv2.VideoCapture(video_path)

    # Load the pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("test")
    # Create the folder if it doesn't exist
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    while True:
        print("ndcddnc")
        ret, frame = cap.read()
        if not ret:
            print("fijvifvifjijvfivjf")
            break
        else:
            print("00000000000000")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the current frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        print(faces)
        for counter, (x, y, w, h) in enumerate(faces):
            print(counter)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 255, 220), 1)
            save(gray, os.path.join(new_path, str(counter)), (x, y, x + w, y + h))

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
video_path = 'video1.mp4'  # Replace with the path to your video file
face_detection_in_video(video_path)
