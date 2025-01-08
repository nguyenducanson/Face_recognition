#
RESIZE_IMAGE_SIZE = 1080

# detection model
YOLO_FACE_MODEL = "weights/yolov8m_face.pt"
DET_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
IMAGE_SIZE = 640

# recognition models
RECOGNITION_MODEL = "minchul/cvlface_adaface_vit_base_kprpe_webface4m"
ALIGN_MODEL = "minchul/cvlface_DFA_mobilenet"
REC_THRESHOLD = 0.5

# database
QDRANT_COLLECTION_NAME = "face_recognition"
EMBEDDING_SIZE = 512
