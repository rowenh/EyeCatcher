import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import settings
import src.video_to_frames
from src.hrnet_api import generate_landmarks
import src.eye_extractor
import numpy as np
import cv2

def get_instances(dataset_path):
    result = []
    for label in os.listdir(dataset_path):
        for instance in os.listdir(dataset_path + "/" + label):
            for file in os.listdir(dataset_path + "/" + label + "/" + instance):
                if not file.endswith(".mp4") and not file.endswith(".avi"):
                    continue
                add_path = dataset_path + "/" + label + "/" + instance
                body_dir = file.replace(".mp4", "").replace(".avi", "").split("_")[0]
                eye = file.replace(".mp4", "").replace(".avi", "").split("_")[1]
                file_extension = file.split(".")[-1]
                result.append([add_path, body_dir, eye, file_extension, label])
                break
    return result

def generate_sample(input_path, output_path, eye):
    if settings.rem_frames <= 0:
        print("Unexpected settings.")
        return

    if os.path.exists(output_path):
        print("Sample already generated.")
        return
    
    original_frame_ids = sorted(os.listdir(input_path + "/frames"), key=lambda x: int(x.replace('.jpg','')))
    total_images = len(original_frame_ids)
    if total_images < settings.rem_frames:
        print("Too few frames available to create samples for " + output_path + ".")
        return

    print("Extracting eye frames...")

    landmarks = src.eye_extractor.get_landmarks(input_path + "/" + settings.landmarks_path, original_frame_ids)
    left_accuracy = 0
    right_accuracy = 0
    acceptable_locations = []
    for i,frame_id in enumerate(original_frame_ids):
        if frame_id not in landmarks:
            continue
        (orig_nose, orig_eye1, orig_eye2) = landmarks[frame_id]
        if orig_nose[2] > settings.landmark_threshold and orig_eye1[2] > settings.landmark_threshold and orig_eye2[2] > settings.landmark_threshold:
            acceptable_locations.append(((orig_nose[0], orig_nose[1]), (orig_eye1[0], orig_eye1[1]), (orig_eye2[0], orig_eye2[1])))
            left_accuracy = left_accuracy + orig_eye1[2]
            right_accuracy = right_accuracy + orig_eye2[2]
    if len(acceptable_locations) == 0:
        print("Landmarks too unreliable to create samples for " + output_path + ".")
        return
    left_accuracy = left_accuracy / len(acceptable_locations)
    right_accuracy = right_accuracy / len(acceptable_locations)

    final_nose_location = (0, 0)
    final_left_eye_location = (0, 0)
    final_right_eye_location = (0, 0)
    for (nose_location, left_eye_location, right_eye_location) in acceptable_locations:
        final_nose_location = (final_nose_location[0] + nose_location[0], final_nose_location[1] + nose_location[1])
        final_left_eye_location = (final_left_eye_location[0] + left_eye_location[0], final_left_eye_location[1] + left_eye_location[1])
        final_right_eye_location = (final_right_eye_location[0] + right_eye_location[0], final_right_eye_location[1] + right_eye_location[1])
    multiplier = 1.0 / len(acceptable_locations)
    final_nose_location = (final_nose_location[0] * multiplier, final_nose_location[1] * multiplier)
    final_left_eye_location = (final_left_eye_location[0] * multiplier, final_left_eye_location[1] * multiplier)
    final_right_eye_location = (final_right_eye_location[0] * multiplier, final_right_eye_location[1] * multiplier)

    # Deal with too much movement
    final_nose = np.array([final_nose_location[0], final_nose_location[1]])
    final_eye1 = np.array([final_left_eye_location[0], final_left_eye_location[1]])
    final_eye2 = np.array([final_right_eye_location[0], final_right_eye_location[1]])
    eye_width = src.eye_extractor.get_eye_width(final_nose, final_eye1, final_eye2)
    movement_threshold = 0.5 * eye_width
    for (nose_location, left_eye_location, right_eye_location) in acceptable_locations:
        nose = np.array([nose_location[0], nose_location[1]])
        eye1 = np.array([left_eye_location[0], left_eye_location[1]])
        eye2 = np.array([right_eye_location[0], right_eye_location[1]])
        if np.linalg.norm(final_eye1 - eye1) > movement_threshold or np.linalg.norm(final_eye2 - eye2) > movement_threshold:
            print("Too much motion to create samples for " + output_path)
            return

    os.mkdir(output_path)

    selected_frame_indices = get_indices(total_images, settings.rem_frames)
    i = -1
    for index in selected_frame_indices:
        i = i + 1
        frame_id = original_frame_ids[index]
        (left_eye_image, right_eye_image) = src.eye_extractor.get_eye_images(cv2.imread(input_path + "/frames/" + frame_id), final_nose_location, final_left_eye_location, final_right_eye_location)
        left_eye_image = cv2.flip(left_eye_image, 1) # Flip to normalize with right eye
        cv2.imwrite(output_path + "/" + str(i) + ".jpg", left_eye_image if eye == "left" else right_eye_image)

def get_indices(input_length, output_length):
    if output_length == 1:
        return [int(input_length / 2)]
    dist = (input_length - 1.0) / (output_length - 1.0)
    indices = []
    for i in range(output_length):
        indices.append(round(i * dist))
    return indices

if __name__ == "__main__":
    fragments = get_instances(os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + "/fragments")
    print(str(len(fragments)) + " fragments found to process.")
    for (path, dir, eye, ext, label) in fragments:
        print("Processing " + path)
        src.video_to_frames.extract_frames(path + "/" + dir + "_" + eye + "." + ext, path + "/frames", settings.default_fps, dir)
    for (path, dir, eye, ext, label) in fragments:
        print("Processing " + path)
        generate_landmarks(path)
    for (path, dir, eye, ext, label) in fragments:
        print("Processing " + path)
        generate_sample(path, os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + "/samples/" + label + "_eye=" + eye + "_" + path.replace("\\", "/").split("/")[-1], eye)
    print("Finished (successfully).")
