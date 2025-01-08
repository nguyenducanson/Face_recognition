import cv2
from ultralytics import YOLO

from constance import (
    IMAGE_SIZE,
    IOU_THRESHOLD,
    DET_THRESHOLD,
    YOLO_FACE_MODEL,
)
from module.utils import crop_bbox


class FaceDetector:
    def __init__(self):
        self.model = YOLO(YOLO_FACE_MODEL)

    def inference(self, img):
        results = self.model.track(
            img,
            iou=IOU_THRESHOLD,
            conf=DET_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=False,
            persist=True,
        )
        img_cp = img.copy()
        crop_boxes = []
        for box in results[0].boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            crop_box = crop_bbox(img, xyxy)
            crop_boxes.append((xyxy, crop_box))

            cv2.rectangle(img_cp, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        return img_cp, crop_boxes
