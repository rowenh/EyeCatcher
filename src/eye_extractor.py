import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import settings
import numpy as np
import cv2
import math

def get_eye_images(frame, nose, eye1, eye2): # For 2D CNN. Left eye, right eye.
    dim = (settings.eye_resolution, settings.eye_resolution)

    nose = np.array([nose[0], nose[1]])
    eye1 = np.array([eye1[0], eye1[1]])
    eye2 = np.array([eye2[0], eye2[1]])

    # Swap for convenience. Note: after swapping, eye1 is right eye; eye2 is left eye.
    tmp = eye1
    eye1 = eye2
    eye2 = tmp

    eye_width = get_eye_width(nose, eye1, eye2)
    eye_to_eye = eye2 - eye1
    angle = 0
    if eye_to_eye[0] != 0. or eye_to_eye[1] != 0.:
        angle = vector_angle([int(eye_to_eye[0]), int(eye_to_eye[1])])
    if eye_to_eye[1] < 0:
        angle = -angle # Rotate other direction.

    right_eye_image = cv2.resize(rotated_crop(frame, (int(eye1[0]), int(eye1[1])), eye_width, angle), dim)
    left_eye_image = cv2.resize(rotated_crop(frame, (int(eye2[0]), int(eye2[1])), eye_width, angle), dim)
    return (left_eye_image, right_eye_image)

def get_landmarks(path, original_frame_ids):
    result = {}
    with open(path) as f:
        body_points = json.load(f)
        frame_ids_sorted = []
        index = -1
        for id in [str(x) for x in sorted([int(x) for x in list(body_points.keys())])]:
            index = index + 1
            if '0' not in list(body_points[id]['persons'].keys()):
                continue
            nose,left_eye,right_eye = body_points[id]['persons']['0'][:3]
            frame_id = original_frame_ids[index]
            result[frame_id] = (nose, left_eye, right_eye)
            frame_ids_sorted.append(frame_id)
    return result

# Helper functions:
def get_eye_width(nose, eye1, eye2):
    # Assume infant pose is always perpendicular to the camera's direction.
    infant_size = np.linalg.norm(nose - (eye1 + (eye2 - eye1) * 0.5))
    eye_width = infant_size * settings.eye_scale
    eye_width = int(eye_width)
    return eye_width

# From: https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/.
def rotated_crop(frame, center, size, angle):
    rect = (center, (size, size), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    crop = crop_rect(frame, rect)
    return crop

# From: https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/.
def crop_rect(frame, rect):
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = frame.shape[0], frame.shape[1]
    frame_rot = cv2.warpAffine(frame, cv2.getRotationMatrix2D(center, angle, 1), (width, height))
    frame_crop = cv2.getRectSubPix(frame_rot, size, center)
    return frame_crop

def vector_angle(vector):
    if (vector[0] * vector[0] + vector[1] * vector[1]) == 0:
        return 0
    return math.degrees(math.acos(vector[0] / math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])))
