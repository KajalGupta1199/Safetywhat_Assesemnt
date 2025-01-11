#Object and Sub-Object Detection using YOLOv5
##Overview
This repository provides an implementation of object and sub-object detection using the YOLOv5 model. The approach includes the detection of main objects (e.g., "Person", "Car") and associated sub-objects (e.g., "Helmet", "Tire") within a hierarchical structure. It also includes modules for optimizing inference speed, modularity, and extensibility.

##Features
Main Object Detection: Detects objects such as "Person", "Car", etc.
Sub-Object Detection: Detects sub-objects like "Helmet", "Tire" within bounding boxes of main objects.
Hierarchical Structure: Organizes detected objects and sub-objects in a hierarchical JSON format.
Real-Time Performance: Processes video frames at a speed of 10-30 FPS using optimization techniques.
Sub-Object Image Cropping: Crops and saves images of detected sub-objects.

##Installation
Step 1: Clone the repository

git clone https://github.com/yourusername/object-subobject-detection.git
cd object-subobject-detection

Step 2: Install dependencies
Make sure you have Python 3.6+ installed. Then, install the required packages using the following commands:

pip install -r requirements.txt
Alternatively, you can install the individual dependencies:


pip install opencv-python torch torchvision pyyaml
pip install yolov5

Step 3: Download YOLOv5 weights
YOLOv5 pre-trained weights are used for the object detection task. You can download them manually or use the following Python code to download the weights:

##Automatically download YOLOv5 weights
python -c "from yolov5 import utils; utils.download('yolov5s.pt')"

Step 4: (Optional) Prepare your own dataset
If you plan to train your own object detection model, you can prepare a custom dataset following the YOLOv5 format. Refer to the official YOLOv5 README for guidance on how to set up your dataset.

##Execution
Step 1: Run Object and Sub-Object Detection on a Video

python detect_objects_and_subobjects.py --video "sample_video.mp4"
This will process the video file sample_video.mp4, detecting both objects and sub-objects in each frame.

Step 2: View Detected Objects and Sub-Objects
The results will be stored in a JSON file named detection_results.json, and a sub-object image will be saved as a PNG file (e.g., subobject_1.png for sub-object 1).

Step 3: Run Real-Time Detection on Camera Feed
For real-time detection from a webcam or external camera:


python detect_objects_and_subobjects.py --webcam
Available Command-Line Arguments:
--video : Path to a video file (e.g., "video.mp4")
--webcam : Use webcam as the input (streams video from the first available camera).
--output-dir : Directory to store output images and JSON results (default is current directory).

##Code Explanation

object Detection and Sub-Object Detection Pipeline
Main Object Detection: Uses YOLOv5 pre-trained models to detect main objects in the input video or image.
Sub-Object Detection: After detecting the main object, a secondary detection process checks for sub-objects within the bounding box of the main object.
Hierarchical Structure: Objects and sub-objects are stored in a hierarchical JSON format:

{
  "object": "Person",
  "id": 1,
  "bbox": [100, 200, 300, 400],
  "subobject": {
    "object": "Helmet",
    "id": 1,
    "bbox": [120, 220, 180, 280]
  }
}

##Code for Cropping and Saving Sub-Object Images

import cv2

def crop_subobject(image, bbox):
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def save_subobject_image(cropped_image, sub_object_id):
    file_path = f"subobject_{sub_object_id}.png"
    cv2.imwrite(file_path, cropped_image)
Inference Speed Optimization

##To optimize for real-time processing, we implement several strategies:

Model Optimization: Use YOLOv5 variants like Tiny YOLO for faster inference.
Multithreading/Concurrency: Split the frame processing into multiple threads to increase throughput.
Batch Processing: Batch multiple frames to process at once (if applicable).
ONNX and TensorRT Optimization: For advanced users, optimize the YOLOv5 model using ONNX for better CPU performance.
Example of monitoring FPS:


import time
import cv2

cap = cv2.VideoCapture('sample_video.mp4')
fps = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    detect_objects(frame)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps}")
    if fps >= 10 and fps <= 30:
        break
cap.release()

Customization and Extensibility

This system is designed to be modular and easily extensible:

##Object Detection Module:
You can replace YOLOv5 with any other detection model.
Sub-Object Detection Module: Add or modify sub-object detectors as required.
Configuration Files: Edit the config.yaml to change object-sub-object pairs to detect.
Example config.yaml:


objects:
  - name: "Person"
    subobjects:
      - name: "Helmet"
      - name: "Gloves"
  - name: "Car"
    subobjects:
      - name: "Tire"
      - name: "Windshield"

##Troubleshooting

Error: "Model not found": Ensure you've downloaded the YOLOv5 weights (yolov5s.pt).
Low FPS: Consider using a faster YOLOv5 variant (e.g., Tiny YOLO) or optimize the model using ONNX.
Video not playing: Ensure your video file path is correct or use --webcam to stream from a camera.
