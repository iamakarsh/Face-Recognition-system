!pip install ultralytics opencv-python matplotlib
from ultralytics import YOLO

model = YOLO("best.pt")
print("Model loaded successfully!")

!pip install ultralytics roboflow opencv-python matplotlib

from ultralytics import YOLO
from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt
import os

# STEP 3 — AUTO-DOWNLOAD ROBOFLOW DATASET

rf = Roboflow(api_key="mIpjj5Gt1z5I4Cxaaxpd")

project = rf.workspace("object-detection-cpbb7").project("face-detection-mik1i-tmgdh")
version = project.version(1)          # Version 1 (dataset version)
dataset = version.download("yolov8")  # Download YOLOv8-ready dataset

print("Dataset downloaded to:", dataset.location)

# STEP 4 — TRAIN YOLOv8 MODEL

model = YOLO("yolov8n.pt")

results = model.train(
    data=dataset.location + "/data.yaml",  # path to YOLO data.yaml
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)

# STEP 5 — VALIDATE MODEL (mAP, Precision, Recall)

metrics = model.val()
print(metrics)

# STEP 6 — RUN PREDICTION ON TEST IMAGES
results = model.predict(
    source=dataset.location + "/test/images",
    save=True,
    conf=0.25
)

print("Predictions saved to: runs/detect/predict/")

# STEP 7 — DISPLAY FIRST PREDICTED IMAGE

predict_folder = "runs/detect/predict"
files = os.listdir(predict_folder)

if len(files) > 0:
    img_path = os.path.join(predict_folder, files[0])
    img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
else:
    print("No predicted images found.")

# OPTIONAL STEP 8 — EXPORT MODEL (ONNX / TFLite / TensorRT)
model.export(format="onnx")
model.export(format="torchscript")
model.export(format="tflite")
model.export(format="engine")

from google.colab import files

uploaded = files.upload()   # upload a test image

results = model.predict(
    source=list(uploaded.keys())[0],
    conf=0.25,
    save=True
)

import cv2
import matplotlib.pyplot as plt
import os

predict_dir = results[0].save_dir
img_name = os.listdir(predict_dir)[0]
img_path = os.path.join(predict_dir, img_name)

img = cv2.imread(img_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

import cv2
import matplotlib.pyplot as plt
import os

predict_dir = results[0].save_dir
img_name = os.listdir(predict_dir)[0]
img_path = os.path.join(predict_dir, img_name)

img = cv2.imread(img_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

model.summary()
