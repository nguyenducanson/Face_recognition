import cv2
import numpy as np
from PIL import Image

from constance import RESIZE_IMAGE_SIZE
from database.qdrant_db import Database
from module import FaceDetector, FaceRecognition
from app.utils import resize_keep_ratio

database = Database()
face_detector = FaceDetector()
face_recognition = FaceRecognition()


# Function to process the camera input
def process_camera_input(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise Exception("Could not open video.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_keep_ratio(frame, RESIZE_IMAGE_SIZE)
        # Detect faces and get bounding boxes
        detected_img, detect_results = face_detector.inference(frame)

        # Match faces with the database
        for coord, face in detect_results:
            face = Image.fromarray(face)
            match_result = face_recognition.inference(face)
            if match_result is not None:
                cv2.putText(
                    detected_img,
                    match_result["user_name"],
                    (int(coord[0]), int(coord[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2
                )

        yield detected_img
    cap.release()


def delete_data():
    pass


def add_data(user_name: str, image: np.ndarray):
    try:
        image = Image.fromarray(image)
        embed = face_recognition.embedding_image(image)
        embed_np = embed.detach().cpu().numpy()[0]
        database.insert_vector((user_name, embed_np))
        return f"Successfully added {user_name}"
    except Exception as e:
        return f"Failed to add {user_name}"
