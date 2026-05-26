import cv2

from app.config import DEFAULT_CONFIDENCE, MODEL_PATH
from app.detector import FaceDetector


def run_webcam(camera_index: int = 0, confidence: float = DEFAULT_CONFIDENCE) -> None:
    detector = FaceDetector(MODEL_PATH)
    detector.configure(confidence=confidence)
    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated, count, _ = detector.draw_detections(frame)
            cv2.putText(
                annotated,
                f"Faces: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLOv8 Face Detection - press q to quit", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
