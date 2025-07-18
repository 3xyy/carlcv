import cv2


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random as rand

from sympy import false

import mpVisualizers as mpVis

# Set up model
modelPath = "MediapipeModels/hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=modelPath)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Constants
NON_PINKY_DIST = 0.3
PINKY_DIST = 0.25
POSSIBLE_FINGERS = {"RPo", "RM", "RR", "RPi", "LPo", "LM", "LR", "LPi"}

fingers = set()
keys_to_press = rand.sample(list(POSSIBLE_FINGERS), rand.randint(1, 3))
completed = False

# Set up camera
cap = cv2.VideoCapture(0)

while True:
    fingers.clear()
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detect_result = detector.detect(mp_image)
    # print(detect_result)

    # TODO: Add code here to detect keypoints

    if completed:
        keys_to_press = rand.sample(list(POSSIBLE_FINGERS), rand.randint(1, 3))
    else:
        print("KEYS TO PRESS:", keys_to_press)
    #Figure out which hand is left and right
    leftHand = None
    rightHand = None
    handedness = detect_result.handedness

    for i in range(len(handedness)):
        if handedness[i][0].category_name == "Left" and leftHand is None:
            leftHand = i
        elif handedness[i][0].category_name == "Right" and rightHand is None:
            rightHand = i

    Rwrist, Rfinger1, Rfinger2, Rfinger3, Rfinger4, Rfinger5 = None, None, None, None, None, None
    if rightHand is not None:
        Rwrist = detect_result.hand_landmarks[rightHand][0].y
        Rfinger1 = abs(Rwrist - detect_result.hand_landmarks[rightHand][4].y)
        Rfinger2 = abs(Rwrist - detect_result.hand_landmarks[rightHand][8].y)
        Rfinger3 = abs(Rwrist - detect_result.hand_landmarks[rightHand][12].y)
        Rfinger4 = abs(Rwrist - detect_result.hand_landmarks[rightHand][16].y)
        Rfinger5 = abs(Rwrist - detect_result.hand_landmarks[rightHand][20].y)

    Lwrist, Lfinger1, Lfinger2, Lfinger3, Lfinger4, Lfinger5 = None, None, None, None, None, None
    if leftHand is not None:
        Lwrist = detect_result.hand_landmarks[leftHand][0].y
        Lfinger1 = abs(Lwrist - detect_result.hand_landmarks[leftHand][4].y)
        Lfinger2 = abs(Lwrist - detect_result.hand_landmarks[leftHand][8].y)
        Lfinger3 = abs(Lwrist - detect_result.hand_landmarks[leftHand][12].y)
        Lfinger4 = abs(Lwrist - detect_result.hand_landmarks[leftHand][16].y)
        Lfinger5 = abs(Lwrist - detect_result.hand_landmarks[leftHand][20].y)

    if Rfinger2 is not None:
        if Rfinger2 < NON_PINKY_DIST:
            fingers.add("RPo")
        else:
            fingers.discard("RPo")
    if Rfinger3 is not None:
        if Rfinger3 < NON_PINKY_DIST:
            fingers.add("RM")
        else:
            fingers.discard("RM")
    if Rfinger4 is not None:
        if Rfinger4 < NON_PINKY_DIST:
            fingers.add("RR")
        else:
            fingers.discard("RR")
    if Rfinger5 is not None:
        if Rfinger5 < PINKY_DIST:
            fingers.add("RPi")
        else:
            fingers.discard("RPi")
    if Lfinger2 is not None:
        if Lfinger2 < NON_PINKY_DIST:
            fingers.add("LPo")
        else:
            fingers.discard("LPo")
    if Lfinger3 is not None:
        if Lfinger3 < NON_PINKY_DIST:
            fingers.add("LM")
        else:
            fingers.discard("LM")
    if Lfinger4 is not None:
        if Lfinger4 < NON_PINKY_DIST:
            fingers.add("LR")
        else:
            fingers.discard("LR")
    if Lfinger5 is not None:
        if Lfinger5 < PINKY_DIST:
            fingers.add("LPi")
        else:
            fingers.discard("LPi")
    if fingers:
        print(fingers)

    all_pressed = True
    for key in keys_to_press:
        if key not in fingers:
            all_pressed = false

    completed = all_pressed

    # print("Right Pointer", Rfinger2, "Right Mid", Rfinger3, "Right Ring", Rfinger4, "Right Pinky", Rfinger5)
    # print("Left Pointer", Lfinger2, "Left Pointer", Lfinger3, "Left Ring", Lfinger4, "Left Pinky", Lfinger5)
    annot_image = mpVis.visualizeHandSkeleton(mp_image.numpy_view(), detect_result)

    vis_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)
    vis_image = cv2.flip(vis_image,1)
    vis_image = cv2.putText(vis_image, str(fingers),
                    (100, 100), cv2.FONT_HERSHEY_DUPLEX,
                    1.5,(255,0,0))

    cv2.imshow("Detected", vis_image)

    x = cv2.waitKey(30)
    ch = chr(x & 0xFF)
    if ch == 'q':
        break

cap.release()

cv2.destroyAllWindows()