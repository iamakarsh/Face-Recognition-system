from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACE_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
MODEL_PATH = FACE_MODEL_PATH
OUTPUT_DIR = PROJECT_ROOT / "static" / "outputs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

DEFAULT_CONFIDENCE = 0.25

MODEL_OPTIONS = {
    "Face Detection": {
        "model": FACE_MODEL_PATH,
        "variants": {"Custom face model": FACE_MODEL_PATH},
        "description": "Your trained YOLOv8 face detector.",
        "metric_label": "Faces detected",
        "short_label": "Face",
    },
    "Person / Object Detection": {
        "model": "yolov8n.pt",
        "variants": {
            "Fast - YOLOv8n": "yolov8n.pt",
            "Balanced - YOLOv8s": "yolov8s.pt",
            "Accurate - YOLOv8m": "yolov8m.pt",
        },
        "description": "Pretrained COCO detector for people and common objects.",
        "metric_label": "Objects detected",
        "short_label": "Object",
    },
    "Pose / Body Keypoints": {
        "model": "yolov8n-pose.pt",
        "variants": {
            "Fast - YOLOv8n pose": "yolov8n-pose.pt",
            "Balanced - YOLOv8s pose": "yolov8s-pose.pt",
            "Accurate - YOLOv8m pose": "yolov8m-pose.pt",
        },
        "description": "Pretrained pose model for shoulders, elbows, wrists, hips, knees, ankles, and more.",
        "metric_label": "People with poses",
        "short_label": "Pose",
    },
}
