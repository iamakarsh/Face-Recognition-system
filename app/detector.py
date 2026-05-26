from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    """Thin inference wrapper around a YOLOv8 model."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = model_path
        if self._is_local_path(model_path) and not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.confidence = 0.25
        self.image_size = 640
        self.max_detections = 100
        self.class_ids: list[int] | None = None

    @staticmethod
    def _is_local_path(model_path: str | Path) -> bool:
        value = str(model_path)
        return any(separator in value for separator in ("\\", "/")) or value.endswith(".pt") and Path(value).parent != Path(".")

    def configure(
        self,
        confidence: float = 0.25,
        image_size: int = 640,
        max_detections: int = 100,
        class_ids: list[int] | None = None,
    ) -> None:
        self.confidence = confidence
        self.image_size = image_size
        self.max_detections = max_detections
        self.class_ids = class_ids

    @property
    def names(self) -> dict[int, str]:
        names = self.model.names
        if isinstance(names, dict):
            return {int(key): value for key, value in names.items()}
        return dict(enumerate(names))

    def class_ids_for_names(self, selected_names: list[str]) -> list[int]:
        selected = set(selected_names)
        return [class_id for class_id, name in self.names.items() if name in selected]

    def predict(self, image: np.ndarray) -> Any:
        return self.model.predict(
            image,
            conf=self.confidence,
            imgsz=self.image_size,
            max_det=self.max_detections,
            classes=self.class_ids,
            verbose=False,
        )[0]

    def draw_detections(self, image: np.ndarray) -> tuple[np.ndarray, int, dict[str, int]]:
        result = self.predict(image)
        annotated = result.plot()
        count = 0 if result.boxes is None else len(result.boxes)
        class_counts = self._class_counts(result)
        return annotated, count, class_counts

    def _class_counts(self, result: Any) -> dict[str, int]:
        if result.boxes is None or result.boxes.cls is None:
            return {}

        counts: dict[str, int] = {}
        for class_id in result.boxes.cls.tolist():
            label = result.names.get(int(class_id), str(int(class_id)))
            counts[label] = counts.get(label, 0) + 1
        return counts

    def detect_image_file(self, image_path: str | Path, output_path: str | Path | None = None) -> tuple[np.ndarray, int, dict[str, int]]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        annotated, count, class_counts = self.draw_detections(image)
        if output_path is not None:
            cv2.imwrite(str(output_path), annotated)
        return annotated, count, class_counts

    def detect_video_file(self, video_path: str | Path, output_path: str | Path) -> tuple[int, dict[str, int]]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        max_detections = 0
        total_class_counts: dict[str, int] = {}
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                annotated, count, class_counts = self.draw_detections(frame)
                max_detections = max(max_detections, count)
                for label, class_count in class_counts.items():
                    total_class_counts[label] = total_class_counts.get(label, 0) + class_count
                writer.write(annotated)
        finally:
            capture.release()
            writer.release()

        return max_detections, total_class_counts


FaceDetector = YoloDetector
