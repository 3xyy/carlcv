"""
File: mpVisualizers.py

This file contains the long, ugly functions that let us visualize Mediapipe's detected faces, hands, bodies, etc.
"""
import math
import cv2
import numpy as np

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
CIRCLE_COLOR = (0, 255, 0)   # green
TEXT_COLOR = (0, 255, 255)  # cyan, remembering that this is applied to an RGB, not a BGR, image
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# ------------------------------------------------------------------------------------------------------
# For Face Detection

def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value):
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not is_valid_normalized_value(normalized_x):
        normalized_x = max(0.0, min(1.0, normalized_x))
    if not is_valid_normalized_value(normalized_y):
        normalized_y = max(0.0, min(1.0, normalized_y))
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def visualizeFaceDetect(image, detection_result):
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualized.
    Returns: Image with bounding boxes.
    """
  
    # Copy the original image and make changes to the copy
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box for each face detected
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw face keypoints for each face detected
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
            cv2.circle(annotated_image, keypoint_px, 3, CIRCLE_COLOR, -1)


        # Draw category label and confidence score as text on bounding box
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image

# ------------------------------------------------------------------------------------------------------
# For Facial Landmarks

def visualizeFacialFeatures(rgb_image, detection_result):
    """
    Draw the face landmark mesh onto a copy of the input RGB image and returns it
    :param rgb_image: an image in RGB format (as a Numpy array)
    :param detection_result: The results of running the face landmarker model
    :return: a copy of rgb_image with face landmark mesh drawn on it
    """
    annotated_image = np.copy(rgb_image)
    face_landmarks_list = detection_result.face_landmarks

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
    """
    Creates a plt bar graph to show how much each blendshape is present in a given image
    :param face_blendshapes: output from the blendshapes model
    :return: 
    """
    # Extract the face blendshapes category names and scores.
    face_blsh_names = [face_blsh_category.category_name for face_blsh_category in face_blendshapes]
    face_blsh_scores = [face_blsh_category.score for face_blsh_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blsh_ranks = range(len(face_blsh_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blsh_ranks, face_blsh_scores, label=[str(x) for x in face_blsh_ranks])
    ax.set_yticks(face_blsh_ranks, face_blsh_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blsh_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------------
# For Hand Landmarks (Skeleton)

def visualizeHandSkeleton(rgb_image, detection_result):
    """
    Draws hand skeleton for each hand visible in an image
    :param rgb_image: An RGB image array
    :param detection_result: The results from the hand landmark detector
    :return: a copy of the input array with the hand skeleton drawn on it, labeled with left or right handedness
    """
    annotated_image = np.copy(rgb_image)

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# ------------------------------------------------------------------------------------------------------
# For Body Pose Landmarks (Skeleton)


def visualizePose(rgb_image, detection_result):
    """
    Draws the pose skeleton on a copy of the input image, based on the data in detection_result
    :param rgb_image: an image in RGB format
    :param detection_result: The results of the pose landmark detector
    :return: a copy of the input image with the pose drawn on it
    """
    annotated_image = np.copy(rgb_image)
    pose_landmarks_list = detection_result.pose_landmarks

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
