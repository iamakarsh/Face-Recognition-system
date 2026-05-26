# Vision Detection Studio

Local YOLOv8 detection project converted from a Colab workflow.

Important: the current `best.pt` model is assumed to be a face detection model, not full face recognition. Recognition of known people requires face embeddings, a known-person database, and matching logic using a library such as DeepFace, FaceNet, ArcFace, or InsightFace.

The app now supports three detection modes:

- Face Detection using your custom `models/best.pt`
- Person / Object Detection using pretrained `yolov8n.pt`
- Pose / Body Keypoints using pretrained `yolov8n-pose.pt`

Ultralytics downloads pretrained weights automatically the first time you select a pretrained mode.

## Accuracy notes

The pretrained object detector is a general COCO model, so it can make mistakes in unusual lighting, blur, partial objects, crowded scenes, or camera angles. It is not trained specifically for your room, phone camera, reels-style clips, or custom body-part labels.

To improve results inside the app:

- Increase confidence to reduce false positives.
- Use **People only** when you care about humans more than every object.
- Try **Balanced** or **Accurate** model quality instead of **Fast**.
- Increase image size for better detection at the cost of speed.
- Use pose mode for wrists, elbows, shoulders, knees, ankles, and body movement.

## Project structure

```text
face-recognition-system/
├── app/
│   ├── main.py
│   ├── camera.py
│   ├── detector.py
│   ├── config.py
│   └── utils.py
├── models/
│   └── best.pt
├── static/
│   └── outputs/
├── templates/
├── training/
├── requirements.txt
├── README.md
└── .env.example
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The model should be located at:

```text
models/best.pt
```

## Run the Streamlit app

```powershell
streamlit run app/main.py
```

The app supports live browser camera detection, camera snapshots, image upload, and video upload detection.

## Run live webcam detection

```powershell
python -m app.camera
```

Press `q` in the webcam window to quit.

If the webcam does not open, check that Windows camera permissions are enabled and that no other app is currently using the camera.

## Environment variables

Copy `.env.example` to `.env` if you need local secrets later.

Do not hardcode Roboflow API keys or other credentials in source code.

## Roadmap

1. Better real-time browser camera support
2. Hand-focused and body-part-specific models
3. Face embedding-based recognition
4. Known person registration
5. Attendance records
6. Unknown face alerts
7. Database, authentication, and deployment
