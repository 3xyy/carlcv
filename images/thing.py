import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mpVisualizers as mpVis

# Set up model
modelPath = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=modelPath)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

def cv_processes():
    ret, frame = cap.read()
    if not ret:
        return

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detect_result = detector.detect(mp_image)

    handedness = detect_result.handedness
    leftHand = None
    rightHand = None

    for i in range(len(handedness)):
        if handedness[i][0].category_name == "Left" and leftHand is None:
            leftHand = i
        elif handedness[i][0].category_name == "Right" and rightHand is None:
            rightHand = i

    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    exclude_indices = set([1,2,3,4])
    all_indices = set(range(21))
    mask_indices = sorted(list(all_indices - exclude_indices))
    circle_radius = 16

    def mask_hand(hand_landmarks):
        # Draw circles at all keypoints except thumb, and connect tips to wrist
        wrist = hand_landmarks[0]
        wx, wy = int(wrist.x * width), int(wrist.y * height)
        wx_m, wy_m = width - wx, wy
        # Draw lines from each tip to wrist for index, middle, ring, pinky
        tip_indices = [8, 12, 16, 20]
        for tip_idx in tip_indices:
            tip = hand_landmarks[tip_idx]
            tx, ty = int(tip.x * width), int(tip.y * height)
            tx_m, ty_m = width - tx, ty
            cv2.line(mask, (tx_m, ty_m), (wx_m, wy_m), 255, circle_radius*2)
        # Draw circles at all non-thumb keypoints
        for idx in mask_indices:
            pt = hand_landmarks[idx]
            x, y = int(pt.x * width), int(pt.y * height)
            x_mirrored = width - x
            cv2.circle(mask, (x_mirrored, y), circle_radius, (255), -1)

    if rightHand is not None:
        mask_hand(detect_result.hand_landmarks[rightHand])
    if leftHand is not None:
        mask_hand(detect_result.hand_landmarks[leftHand])

    # Mirror the frame horizontally to match the mask
    frame_mirrored = cv2.flip(frame, 1)
    # Show only the masked area (set everything else to black)
    hands_only = np.zeros_like(frame_mirrored)
    hands_only[mask == 255] = frame_mirrored[mask == 255]
    cv2.imshow("Hands Only", hands_only)

    # Detected window (skeleton overlay)
    annot_image = mpVis.visualizeHandSkeleton(mp_image.numpy_view(), detect_result)
    vis_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)
    vis_image = cv2.flip(vis_image, 1)
    cv2.imshow("Detected", vis_image)

    # Exit on 'q' key
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

while True:
    cv_processes()