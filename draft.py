import cv2
import numpy as np
import pygame

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import mpVisualizers as mpVis

# Set up model
modelPath = "MediapipeModels/hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=modelPath)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Set up camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detect_result = detector.detect(mp_image)
    # print(detect_result)

    # TODO: Add code here to detect keypoints
    

    annot_image = mpVis.visualizeHandSkeleton(mp_image.numpy_view(), detect_result)
    vis_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Detected", vis_image)

    x = cv2.waitKey(30)
    ch = chr(x & 0xFF)
    if ch == 'q':
        break

cap.release()

cv2.destroyAllWindows()


