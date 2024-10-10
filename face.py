import cv2
import face_recognition
import sys
sys.path.append('/path/to/your/face_recognition_models folder')

import face_recognition_models

video_capture = cv2.VideoCapture(0)

# Load known images and encode them
known_image = face_recognition.load_image_file("person2.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
known_face_encodings = [known_encoding]
known_face_names = ["Shiya"]
# Add more known faces
known_image_2 = face_recognition.load_image_file("person.jpg")
known_encoding_2 = face_recognition.face_encodings(known_image_2)[0]
known_face_encodings.append(known_encoding_2)
known_face_names.append("Shashank")
# Add more known faces
known_image_2 = face_recognition.load_image_file("person3.jpg")
known_encoding_2 = face_recognition.face_encodings(known_image_2)[0]
known_face_encodings.append(known_encoding_2)
known_face_names.append("Kartik")



while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)

    face_encodings = []

    if len(face_locations) > 0:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
